import glob
import math
import cv2
import torch
import yt_dlp
from yt_dlp.utils import download_range_func
import os
import re
import ffmpeg
from super_gradients.training import models
from super_gradients.training.utils.visualization.pose_estimation import PoseVisualization
from tensorflow.keras.models import load_model # type: ignore
import numpy as np

video_url = "https://www.youtube.com/watch?v=PRscy-DzQDA"
start_time = 0
end_time = 15
skier_number = 0
start_frame = 0
subframe_length = 30


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


def download_video(link, start_time, end_time, download_path):
    """
    Download a video with the best video quality from YouTube using yt-dlp.
    """
    cur_dir = os.getcwd()
    youtube_dl_options = {
        'download_ranges': download_range_func(None, [(start_time, end_time)]),
        'force_keyframes_at_cuts': True,
        'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
        'format_sort': ['res:1080', 'ext:mp4:m4a'],
        'encoding': 'utf-8',
        'outtmpl': os.path.join(cur_dir, download_path),
        'quiet': True,
        'no_warnings': True,
    }
    try:
        with yt_dlp.YoutubeDL(youtube_dl_options) as ydl:
            ydl.download([link])
    except Exception as e:
        print(f"Error downloading video: {str(e)}")


def reduce_fps(video_path, output_path, new_fps):
    """
    Reduce the FPS of the video to 30 for consistency
    """
    video = ffmpeg.input(video_path)
    video = video.filter('fps', fps=new_fps, round='up')
    ffmpeg.output(video, output_path).run()


def process_single_image(image_prediction, video_name, frame_num):
    """
    Process a single image and save the pose data.

    Returns the image with pose data drawn on it.
    """
    image = image_prediction.image
    pose_data = image_prediction.prediction
    poses = pose_data.poses.tolist()
    boxes = pose_data.bboxes_xyxy.tolist()

    with open(f"test/pose_data_{video_name}.txt", "a") as f:
        f.write(str(pose_data.poses.tolist())+"\n")
        f.write(str(pose_data.bboxes_xyxy.tolist())+"\n")
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


def create_superimposed_video_from_images(images, output_video_name="output_video.mp4", fps=30.0):
    """
    Create a video from a list of images.
    """
    frame_height, frame_width, layers = images[0].shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_video_name, fourcc, fps, (frame_width, frame_height))

    for frame in images:
        video.write(frame)

    video.release()
    cv2.destroyAllWindows()


def skier_num(poses, boxes, *args):
    """
    Return the skier number with the least error from the previous frame
    """

    # If the previous frame is empty, return the first skier
    skier, previous_pose, previous_box, frames_skipped = args
    if previous_pose == []:
        previous_pose = poses[skier]
        previous_box = boxes[skier]
        return (skier, previous_pose, previous_box, frames_skipped)
    
    errors = []
    for pose in poses:
        error = 0
        for i in range(17):
            error += math.sqrt((pose[i][0] - previous_pose[i][0])**2 + (pose[i][1] - previous_pose[i][1])**2)
        errors.append(error)
    skier = errors.index(min(errors))
    box_size = max(previous_box[2] - previous_box[0], previous_box[3] - previous_box[1])
    if errors[skier] / 17 > box_size / 5 * (frames_skipped + 1):
        skier = -1
        frames_skipped += 1
        return (-1, previous_pose, previous_box, frames_skipped)
    frames_skipped = 0
    previous_pose = poses[skier]
    previous_box = boxes[skier]
    return (skier, previous_pose, previous_box, frames_skipped)


def track_skier(video_name, skier_number, start_frame):
    """
    Track the skier in the video and write the pose and box data to a file
    """
    skier = skier_number
    previous_pose = []
    previous_box = []
    frames_skipped = 0

    # Read in the pose data
    with open(f"test/pose_data_{video_name}.txt", "r") as f:
        data = f.read().split("\n\n")
        all_poses = []
        all_boxes = []
        for frame in data[1:]:
            if frame == "":
                continue
            poses = eval(frame.split("\n")[0])
            boxes = eval(frame.split("\n")[1])
            all_poses.append(poses)
            all_boxes.append(boxes)
            
    with open(f"test/skier_tracked_{video_name}.txt", "w") as dataset_f:
        for frame in range(start_frame, len(all_poses)):
            # If there is no one in the current frame, skip it
            if all_poses[frame] == []:
                frames_skipped += 1
            else:
                args = skier_num(all_poses[frame], all_boxes[frame], skier, previous_pose, previous_box, frames_skipped)
                skier, previous_pose, previous_box, frames_skipped = args
            
            # If the skier is in the current frame, write the pose and box to the file
            if skier != -1 and skier < len(all_poses[frame]):
                dataset_f.write(str(all_poses[frame][skier]))
                dataset_f.write("\n")
                dataset_f.write(str(all_boxes[frame][skier]))
                dataset_f.write("\n\n")
            # Otherwise, write empty lists
            else:
                dataset_f.write("[]")
                dataset_f.write("\n")
                dataset_f.write("[]")
                dataset_f.write("\n\n")
        dataset_f.close()


def get_tracked_skier(video_name):
    """
    Returns the tracked skier data from the file
    """
    with open(f"test/skier_tracked_{video_name}.txt", "r") as f:
        data = f.read().split("\n\n")
        poses = []
        boxes = []
        for frame in data:
            if frame == "":
                continue
            pose = eval(frame.split("\n")[0])
            box = eval(frame.split("\n")[1])
            poses.append(pose)
            boxes.append(box)
        return poses, boxes


def create_tracked_video_from_images(video_name, poses, boxes, output_filename="output_video.mp4", fps=30.0):
    """
    Create a video from a list of images and the tracked skier data.
    """

    cap = cv2.VideoCapture(f"test/{video_name}.mp4")
    cap.read()
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    frame = 0

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
        
        scale = frame_width / 1000

        pose = poses[frame]
        box = boxes[frame]

        EDGE_LINKS = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 6], [5, 7], [5, 11], [6, 8], [6, 12], [7, 9], [8, 10], [11, 12], [11, 13], [12, 14], [13, 15], [14, 16]]
        EDGE_COLORS = [[214, 39, 40], [148, 103, 189], [44, 160, 44], [140, 86, 75], [227, 119, 194], [127, 127, 127], [188, 189, 34], [127, 127, 127], [188, 189, 34], [140, 86, 75], [23, 190, 207], [227, 119, 194], [31, 119, 180], [255, 127, 14], [148, 103, 189], [255, 127, 14], [214, 39, 40], [31, 119, 180], [44, 160, 44]]
        KEYPOINT_COLORS = [[148, 103, 189], [31, 119, 180], [148, 103, 189], [31, 119, 180], [148, 103, 189], [31, 119, 180], [148, 103, 189], [31, 119, 180], [148, 103, 189], [31, 119, 180], [148, 103, 189], [31, 119, 180], [148, 103, 189], [31, 119, 180], [148, 103, 189], [31, 119, 180], [148, 103, 189]]
        
        BOUNDING_BOX_COLOR = (255, 255, 255)

        for idx, edge in enumerate(EDGE_LINKS):
            start, end = edge
            if start < len(pose) and end < len(pose):
                x_coords = [int(pose[start][0]), int(pose[end][0])]
                y_coords = [int(pose[start][1]), int(pose[end][1])]
                confidence = (pose[start][2] + pose[end][2]) / 2
                cv2.line(image, (x_coords[0], y_coords[0]), (x_coords[1], y_coords[1]), EDGE_COLORS[idx], thickness=int(scale)+1)

        # keypoints
        for idx, keypoint in enumerate(pose):
            cv2.circle(image, (int(keypoint[0]), int(keypoint[1])), radius=int(scale*3), color=KEYPOINT_COLORS[idx], thickness=(int(scale)+1)//2)
    
        # bounding box
        if len(box) == 4:
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), BOUNDING_BOX_COLOR, thickness=2)
        frame += 1

        video.write(image)

    video.release()
    cv2.destroyAllWindows()


def normalize_coordinates(keypoint, box):
    x_normalized, y_normalized = 0, 0
    if box[2] - box[0] > box[3] - box[1]:
        x_normalized = 1000 * (keypoint[0] - box[0]) / (box[2] - box[0])
        y_normalized = 1000 * (box[3] - box[1]) / (box[2] - box[0]) * (keypoint[1] - box[1]) / (box[3] - box[1])
        y_normalized += 1000 * (1 - (box[3] - box[1]) / (box[2] - box[0])) / 2
    else:
        x_normalized = 1000 * (box[2] - box[0]) / (box[3] - box[1]) * (keypoint[0] - box[0]) / (box[2] - box[0])
        x_normalized += 1000 * (1 - (box[2] - box[0]) / (box[3] - box[1])) / 2
        y_normalized = 1000 * (keypoint[1] - box[1]) / (box[3] - box[1])
    return (x_normalized, y_normalized)


def get_normalized_coordinates(video_name):
    """
    Returns the normalized coordinates of the keypoint based on the bounding box.
    """
    with open(f"test/skier_tracked_{video_name}.txt", "r") as f:
        data = f.read().split("\n\n")
        all_normalized_coordinates = []
        for frame in data:
            if frame == "":
                continue
            pose = eval(frame.split("\n")[0])
            box = eval(frame.split("\n")[1])
            normalized_coordinates = []
            for keypoint in pose:
                coordinates = normalize_coordinates(keypoint, box)
                normalized_coordinates.append(coordinates)
            all_normalized_coordinates.append(normalized_coordinates)
        return all_normalized_coordinates


def get_rating(points):
    """
    Returns the rating of the skier based on the points
    """
    flip = 1000 - points
    return flip / 10


def create_final_video_from_images(video_name, predictions, poses, boxes, output_filename="output_video.mp4", fps=30.0):
    """
    Create a video from a list of images and add predictions to the video.
    """

    cap = cv2.VideoCapture(f"static/videos/{video_name}.mp4")
    cap.read()
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    frame = 0
    last_valid_rating = [rating for rating in predictions[::-1] if rating != "skier not found"][0]

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
        if frame >= len(predictions):
            break

        rating = predictions[frame]
        if rating == "skier not found":
            predictions[frame] = last_valid_rating
            rating = last_valid_rating
            color = (0, 255, 255)
        else:
            last_valid_rating = rating
            if rating > predictions[frame - 1] and predictions[frame - 1] > predictions[frame - 2] and predictions[frame - 2] > predictions[frame - 3]:
                color = (0, 255, 0)
            elif rating < predictions[frame - 1] and predictions[frame - 1] < predictions[frame - 2] and predictions[frame - 2] < predictions[frame - 3]:
                color = (0, 0, 255)
            else:
                color = (0, 255, 255)
        
        scale = frame_width / 1000
        height_scale = frame_height / 1000
        org = (int(50 * scale), int(100 * height_scale))
        fontscale = scale
        thickness = int(scale * 2)
        cv2.putText(image, f"Rating: {rating}", org, cv2.FONT_HERSHEY_SIMPLEX, fontscale, color, thickness)

        pose = poses[frame]
        box = boxes[frame]

        EDGE_LINKS = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 6], [5, 7], [5, 11], [6, 8], [6, 12], [7, 9], [8, 10], [11, 12], [11, 13], [12, 14], [13, 15], [14, 16]]
        EDGE_COLORS = [[214, 39, 40], [148, 103, 189], [44, 160, 44], [140, 86, 75], [227, 119, 194], [127, 127, 127], [188, 189, 34], [127, 127, 127], [188, 189, 34], [140, 86, 75], [23, 190, 207], [227, 119, 194], [31, 119, 180], [255, 127, 14], [148, 103, 189], [255, 127, 14], [214, 39, 40], [31, 119, 180], [44, 160, 44]]
        KEYPOINT_COLORS = [[148, 103, 189], [31, 119, 180], [148, 103, 189], [31, 119, 180], [148, 103, 189], [31, 119, 180], [148, 103, 189], [31, 119, 180], [148, 103, 189], [31, 119, 180], [148, 103, 189], [31, 119, 180], [148, 103, 189], [31, 119, 180], [148, 103, 189], [31, 119, 180], [148, 103, 189]]
        
        for idx, edge in enumerate(EDGE_LINKS):
            start, end = edge
            if start < len(pose) and end < len(pose):
                x_coords = [int(pose[start][0]), int(pose[end][0])]
                y_coords = [int(pose[start][1]), int(pose[end][1])]
                confidence = (pose[start][2] + pose[end][2]) / 2
                cv2.line(image, (x_coords[0], y_coords[0]), (x_coords[1], y_coords[1]), EDGE_COLORS[idx], thickness=int(scale)+1)

        # keypoints
        for idx, keypoint in enumerate(pose):
            cv2.circle(image, (int(keypoint[0]), int(keypoint[1])), radius=int(scale*3), color=KEYPOINT_COLORS[idx], thickness=(int(scale)+1)//2)
    
        # bounding box
        if len(box) == 4:
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness=2)
        frame += 1

        video.write(image)

    video.release()
    cv2.destroyAllWindows()


def encode_video(input_path, output_path):
    # Make sure the video is in the correct format
    ffmpeg.input(input_path).output(
        output_path,
        vcodec="libx264",
        crf=18,
        preset="fast",
        acodec="aac",
        audio_bitrate="128k",
    ).run()


def main():
    video_id = extract_video_id(video_url)
    video_name = f"{video_id}_{start_time}_{end_time}"

    files = glob.glob('test/*')
    for f in files:
        os.remove(f)

    # Download the video
    download_video(video_url, start_time, end_time, f"test/tmp_{video_name}.mp4")
    reduce_fps(f"test/tmp_{video_name}.mp4", f"test/{video_name}.mp4", 30.0)
    os.remove(f"test/tmp_{video_name}.mp4")

    # Get pose data
    with open(f"test/pose_data_{video_name}.txt", "w") as f:
        f.write("")
        f.close()
    
    # Make sure that YOLO-NAS-POSE is using the best available hardware
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    yolo_nas_pose = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose").to(device)
    predictions = yolo_nas_pose.predict(f"test/{video_name}.mp4", conf=.1)
    processed_images = [
        process_single_image(image_prediction, video_name, frame_num)
        for frame_num, image_prediction in enumerate(predictions._images_prediction_gen)
    ]
    if os.path.exists(f'test/yolo_{video_name}.mp4'):
        os.remove(f'test/yolo_{video_name}.mp4')
    create_superimposed_video_from_images(processed_images, f'test/tmp_superimposed_{video_name}.mp4', fps=30.0)
    encode_video(f'test/tmp_superimposed_{video_name}.mp4', f'test/superimposed_{video_name}.mp4')
    os.remove(f'test/tmp_superimposed_{video_name}.mp4')

    # Track skier
    track_skier(video_name, skier_number, start_frame)
    poses, boxes = get_tracked_skier(video_name)

    if os.path.exists(f"test/skier_tracked_{video_name}.mp4"):
        os.remove(f"test/skier_tracked_{video_name}.mp4")
    create_tracked_video_from_images(video_name, poses, boxes, f"test/tmp_tracked_{video_name}.mp4", fps=30.0)
    encode_video(f"test/tmp_tracked_{video_name}.mp4", f"test/tracked_{video_name}.mp4")
    os.remove(f"test/tmp_tracked_{video_name}.mp4")

    # Get normalized coordinates
    normalized_coordinates = get_normalized_coordinates(video_name)

    # Load the Keras model
    model = load_model('models/cnn_model.keras', compile=False)

    NOT_FOUND = "skier not found"
    predictions = [NOT_FOUND] * subframe_length
    raw_predictions = [NOT_FOUND] * subframe_length
    for subframe in range(subframe_length, len(normalized_coordinates)):
        subframes = normalized_coordinates[subframe - subframe_length : subframe]
        if [] in subframes:
            predictions.append(NOT_FOUND)
            continue
        subframes = np.array(subframes)
        subframes = subframes.reshape((subframe_length, 34))
        subframes = np.expand_dims(subframes, axis=0)
        prediction = model.predict(subframes, verbose=None)
        raw_predictions.append(prediction[0][0])
        rating = get_rating(prediction[0][0])
        rating = round(rating, 2)
        predictions.append(rating)
    print(f"Raw predictions: {raw_predictions}")
    print(f"Predictions: {predictions}")
    
    # Create video with predictions
    if os.path.exists(f"test/output_{video_name}.mp4"):
        os.remove(f"test/output_{video_name}.mp4")
    create_final_video_from_images(video_name, predictions, poses, boxes, f"test/tmp_output_{video_name}.mp4", fps=10.0)
    encode_video(f"test/tmp_output_{video_name}.mp4", f"test/output_{video_name}.mp4")
    os.remove(f"test/tmp_output_{video_name}.mp4")


if __name__ == '__main__':
    main()