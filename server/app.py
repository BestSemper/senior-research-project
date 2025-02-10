import math
from flask import Flask, request, render_template, redirect, url_for
import yt_dlp
from yt_dlp.utils import download_range_func
import os
import glob
import re
import ffmpeg
import cv2
from super_gradients.training import models
from super_gradients.training.utils.visualization.pose_estimation import PoseVisualization
import numpy

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

def extract_video_id(youtube_url):
    # Standard format: https://www.youtube.com/watch?v=VIDEO_ID
    match = re.search(r'v=([A-Za-z0-9_-]+)', youtube_url)
    if match:
        return match.group(1)

    # Shortened format: https://youtu.be/VIDEO_ID
    match = re.search(r'youtu\.be/([A-Za-z0-9_-]+)', youtube_url)
    if match:
        return match.group(1)

    # YouTube shorts format: https://www.youtube.com/shorts/VIDEO_ID
    match = re.search(r'/shorts/([A-Za-z0-9_-]+)', youtube_url)
    if match:
        return match.group(1)

    raise ValueError('Invalid YouTube URL')

def download_clip(link, id, start_time, end_time):
    cur_dir = os.getcwd()
    youtube_dl_options = {
        'download_ranges': download_range_func(None, [(start_time, end_time)]),
        'force_keyframes_at_cuts': True,
        'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
        'format_sort': ['res:1080', 'ext:mp4:m4a'],
        'encoding': 'utf-8',
        'outtmpl': os.path.join(cur_dir, f'static/videos/{id}_{start_time}-{end_time}_tmp.mp4'),
        'quiet': True,
        'no_warnings': True,
    }
    try:
        with yt_dlp.YoutubeDL(youtube_dl_options) as ydl:
            ydl.download([link])
    except Exception as e:
        print(f"Error downloading video: {str(e)}")

def reduce_fps(video_path, output_path, new_fps):
    video = ffmpeg.input(video_path)
    video = video.filter('fps', fps=new_fps, round='up')
    ffmpeg.output(video, output_path).run()

@app.route('/download_video', methods=['POST'])
def download_video():
    # Remove videos from previous sessions to prevent caching errors
    files = glob.glob('static/videos/*')
    for f in files:
        os.remove(f)
    files = glob.glob('static/output/*')
    for f in files:
        os.remove(f)

    # Extract video ID from YouTube URL
    video_url = request.form.get('video_url')
    video_id = extract_video_id(video_url)
    start_time = int(request.form.get('start_time'))
    end_time = int(request.form.get('end_time'))
    video_name = f'{video_id}_{start_time}-{end_time}'
    tmp_video_name = f'{video_id}_{start_time}-{end_time}_tmp'

    # Download video clip and reduce FPS to 30
    download_clip(video_url, video_id, start_time, end_time)
    reduce_fps(f'static/videos/{tmp_video_name}.mp4', f'static/videos/{video_name}.mp4', 30)
    os.remove(f'static/videos/{tmp_video_name}.mp4')

    return redirect(url_for('downloaded', video_name=video_name))

@app.route('/downloaded')
def downloaded():
    video_name = request.args.get('video_name')
    return render_template('downloaded.html', video_name=video_name)

def superimpose_single_image(image_prediction, video_name, frame_num):
    image = image_prediction.image
    pose_data = image_prediction.prediction
    poses = pose_data.poses.tolist()
    boxes = pose_data.bboxes_xyxy.tolist()

    with open(f"static/output/{video_name}.txt", "a") as f:
        f.write(str(poses)+"\n")
        f.write(str(boxes)+"\n")
        f.write("\n")
        f.close()
    
    skeleton_image = PoseVisualization.draw_poses(
        image=image,
        poses=pose_data.poses,
        boxes=pose_data.bboxes_xyxy,
        scores=pose_data.scores,
        is_crowd=None,
        edge_links=pose_data.edge_links,
        edge_colors=pose_data.edge_colors,
        keypoint_colors=pose_data.keypoint_colors,
        joint_thickness=2,
        box_thickness=2,
        keypoint_radius=5
    )

    if frame_num < 90:
        for skier_num in range(len(poses)):
            cv2.putText(
                skeleton_image, f'number: {skier_num}', (int(boxes[skier_num][0]), int(boxes[skier_num][1]) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA
            )
        cv2.putText(skeleton_image, f'frame: {frame_num}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    
    return skeleton_image

def create_video_from_frames(frames, output_filename='output_video.mp4', fps=30.0):
    # Determine the width and height from the first frame
    frame_height, frame_width, layers = frames[0].shape

    # Define the codec for .mp4 format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    # Write each frame to the video
    for frame in frames:
        video.write(frame)

    # Close and release everything
    video.release()
    cv2.destroyAllWindows()

def encode_video(input_path, output_path):
    ffmpeg.input(input_path).output(output_path, vcodec='libx264', crf=18, preset='fast', acodec='aac', audio_bitrate='128k').run()

@app.route('/superimpose', methods=['POST'])
def superimpose():
    video_name = request.form.get('video_name')
    video = cv2.VideoCapture(f"static/videos/{video_name}.mp4")
    with open(f"static/output/{video_name}.txt", "w") as f:
        f.write(str([int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))])+"\n\n")
        f.close()
    yolo_nas_pose = models.get('yolo_nas_pose_l', pretrained_weights='coco_pose').cuda()
    predictions = yolo_nas_pose.to('cuda').predict(f'static/videos/{video_name}.mp4', conf=.3)
    processed_frames = [superimpose_single_image(image_prediction, video_name, frame_num) for frame_num, image_prediction in enumerate(predictions._images_prediction_gen)]
    files = glob.glob('static/videos/superimposed_*.mp4')
    for previous_f in files:
        os.remove(previous_f)
    create_video_from_frames(processed_frames, f'static/videos/tmp_superimposed_{video_name}.mp4', fps=30.0)
    encode_video(f'static/videos/tmp_superimposed_{video_name}.mp4', f'static/videos/superimposed_{video_name}.mp4')
    os.remove(f'static/videos/tmp_superimposed_{video_name}.mp4')
    return redirect(url_for('superimposed', video_name=video_name))

@app.route('/superimposed')
def superimposed():
    video_name = request.args.get('video_name')
    return render_template('superimposed.html', video_name=video_name)

skier = 0
skier_through_video = []
previous_pose = []
previous_box = []
frames_skipped = 0

def initial_conditions(all_poses, all_boxes):
    global skier, previous_pose, previous_box, frames_skipped
    max_frames = []
    initial_skiers = []
    start_frames = []
    for start_frame in range(30):
        for initial_skier in range(len(all_poses[start_frame])):
            tmp_skier = initial_skier
            previous_pose = []
            previous_box = []
            frames_skipped = 0
            consecutive_frames = 0
            for frame in range(start_frame, len(all_poses)):
                tmp_skier = skier_num(all_poses[frame], all_boxes[frame])
                if tmp_skier == -1:
                    break
                consecutive_frames += 1
            max_frames.append(consecutive_frames)
            initial_skiers.append(initial_skier)
            start_frames.append(start_frame)
    skier_return = initial_skiers[max_frames.index(max(max_frames))]
    start_frame_return = start_frames[max_frames.index(max(max_frames))]
    print(max_frames)
    print(skier_return)
    print(start_frame_return)
    return skier_return, start_frame_return

def skier_num(poses, boxes):
    global skier, previous_pose, previous_box, frames_skipped
    if previous_pose == []:
        previous_pose = poses[skier]
        previous_box = boxes[skier]
        return skier
    errors = []
    for pose in poses:
        error = 0
        for i in range(17):
            error += math.sqrt((pose[i][0] - previous_pose[i][0])**2 + (pose[i][1] - previous_pose[i][1])**2)
        errors.append(error)
    skier = errors.index(min(errors))
    box_size = max(previous_box[2] - previous_box[0], previous_box[3] - previous_box[1])
    if errors[skier]/17 > box_size/5*(frames_skipped+1):
        skier = -1
        frames_skipped += 1
        return -1
    frames_skipped = 0
    previous_pose = poses[skier]
    previous_box = boxes[skier]
    return skier

def track_single_image(dimensions, poses, boxes):
    global frames_skipped, skier_through_video
    skeleton_image = numpy.zeros((dimensions[0], dimensions[1], 3), dtype=numpy.uint8)

    if poses == []:
        frames_skipped += 1
        return skeleton_image

    skier_num(poses, boxes)
    skier_through_video.append(skier)

    if skier == -1:
        return skeleton_image

    pose = poses[skier]
    box = boxes[skier]

    EDGE_LINKS = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 6], [5, 7], [5, 11], [6, 8], [6, 12], [7, 9], [8, 10], [11, 12], [11, 13], [12, 14], [13, 15], [14, 16]]
    EDGE_COLORS = [[214, 39, 40], [148, 103, 189], [44, 160, 44], [140, 86, 75], [227, 119, 194], [127, 127, 127], [188, 189, 34], [127, 127, 127], [188, 189, 34], [140, 86, 75], [23, 190, 207], [227, 119, 194], [31, 119, 180], [255, 127, 14], [148, 103, 189], [255, 127, 14], [214, 39, 40], [31, 119, 180], [44, 160, 44]]
    KEYPOINT_COLORS = [[148, 103, 189], [31, 119, 180], [148, 103, 189], [31, 119, 180], [148, 103, 189], [31, 119, 180], [148, 103, 189], [31, 119, 180], [148, 103, 189], [31, 119, 180], [148, 103, 189], [31, 119, 180], [148, 103, 189], [31, 119, 180], [148, 103, 189], [31, 119, 180], [148, 103, 189]]

    BOX_COLOR = (0, 255, 0)

    for idx, edge in enumerate(EDGE_LINKS):
        start, end = edge
        if start < len(pose) and end < len(pose):
            x_coords = [int(pose[start][0]), int(pose[end][0])]
            y_coords = [int(pose[start][1]), int(pose[end][1])]
            cv2.line(skeleton_image, (x_coords[0], y_coords[0]), (x_coords[1], y_coords[1]), EDGE_COLORS[idx], thickness=2)

    # keypoints
    for idx, keypoint in enumerate(pose):
        cv2.circle(skeleton_image, (int(keypoint[0]), int(keypoint[1])), radius=5, color=KEYPOINT_COLORS[idx], thickness=-1)

    # bounding box
    if len(box) == 4:
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(skeleton_image, (x_min, y_min), (x_max, y_max), BOX_COLOR, thickness=2)

    return skeleton_image

@app.route('/track_skier', methods=['POST'])
def track_skier():
    global skier, previous_pose, previous_box, frames_skipped
    video_name = request.form.get('video_name')
    skier = int(request.form.get('skier_number'))
    start_frame = int(request.form.get('start_frame'))
    with open(f"static/output/{video_name}.txt", "r") as f:
        data = f.read().split("\n\n")
        dimensions = eval(data[0])
        all_poses = []
        all_boxes = []
        for frame in data[1:]:
            if frame == "":
                continue
            poses = eval(frame.split("\n")[0])
            boxes = eval(frame.split("\n")[1])
            all_poses.append(poses)
            all_boxes.append(boxes)
        previous_pose = []
        previous_box = []
        frames_skipped = 0
        processed_frames = []
        skier_tracked = []
        for frame in range(start_frame, len(all_poses)):
            processed_frames.append(track_single_image(dimensions, all_poses[frame], all_boxes[frame]))
            skier_pose = poses[skier] if skier != -1 and (skier < len(poses)) else []
            skier_tracked.append(skier_pose)
        with open(f"static/output/tracked_{video_name}.txt", "w") as dataset_f:
            dataset_f.write(str([int(dimensions[0]), int(dimensions[1])])+"\n")
            for frame in skier_tracked:
                dataset_f.write(str(frame)+"\n")
            dataset_f.close()
        files = glob.glob('static/videos/tracked_*.mp4')
        for previous_f in files:
            os.remove(previous_f)
        create_video_from_frames(processed_frames, f"static/videos/tmp_tracked_{video_name}.mp4")
        encode_video(f"static/videos/tmp_tracked_{video_name}.mp4", f"static/videos/tracked_{video_name}.mp4")
        os.remove(f"static/videos/tmp_tracked_{video_name}.mp4")
        f.close()
    return redirect(url_for('skier_tracked', video_name=video_name))

@app.route('/skier_tracked')
def skier_tracked():
    video_name = request.args.get('video_name')
    score = 0
    return render_template('skier_tracked.html', video_name=video_name, score=score)

def superimpose_tracked_skier(dimensions, poses, boxes):
    skeleton_image = numpy.zeros((dimensions[0], dimensions[1], 3), dtype=numpy.uint8)

    pose = poses[skier]
    box = boxes[skier]

    EDGE_LINKS = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 6], [5, 7], [5, 11], [6, 8], [6, 12], [7, 9], [8, 10], [11, 12], [11, 13], [12, 14], [13, 15], [14, 16]]
    EDGE_COLORS = [[214, 39, 40], [148, 103, 189], [44, 160, 44], [140, 86, 75], [227, 119, 194], [127, 127, 127], [188, 189, 34], [127, 127, 127], [188, 189, 34], [140, 86, 75], [23, 190, 207], [227, 119, 194], [31, 119, 180], [255, 127, 14], [148, 103, 189], [255, 127, 14], [214, 39, 40], [31, 119, 180], [44, 160, 44]]
    KEYPOINT_COLORS = [[148, 103, 189], [31, 119, 180], [148, 103, 189], [31, 119, 180], [148, 103, 189], [31, 119, 180], [148, 103, 189], [31, 119, 180], [148, 103, 189], [31, 119, 180], [148, 103, 189], [31, 119, 180], [148, 103, 189], [31, 119, 180], [148, 103, 189], [31, 119, 180], [148, 103, 189]]

    BOX_COLOR = (0, 255, 0)

    for idx, edge in enumerate(EDGE_LINKS):
        start, end = edge
        if start < len(pose) and end < len(pose):
            x_coords = [int(pose[start][0]), int(pose[end][0])]
            y_coords = [int(pose[start][1]), int(pose[end][1])]
            cv2.line(skeleton_image, (x_coords[0], y_coords[0]), (x_coords[1], y_coords[1]), EDGE_COLORS[idx], thickness=2)

    # keypoints
    for idx, keypoint in enumerate(pose):
        cv2.circle(skeleton_image, (int(keypoint[0]), int(keypoint[1])), radius=5, color=KEYPOINT_COLORS[idx], thickness=-1)

    # bounding box
    if len(box) == 4:
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(skeleton_image, (x_min, y_min), (x_max, y_max), BOX_COLOR, thickness=2)

    return skeleton_image

@app.route('/results', methods=['POST'])
def results():
    video_name = request.form.get('video_name')
    score = 9
    return render_template('results.html', video_name=video_name, score=score)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)