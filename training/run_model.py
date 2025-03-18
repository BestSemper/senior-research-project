import glob
import math
import yt_dlp
from yt_dlp.utils import download_range_func
import os
import re
import ffmpeg
from super_gradients.training import models
from keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import numpy as np

video_url = "https://www.youtube.com/watch?v=bSXyTlDcXPk"
start_time = 54
end_time = 66
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

def download_video(link, id, start_time, end_time, download_path):
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
    video = ffmpeg.input(video_path)
    video = video.filter('fps', fps=new_fps, round='up')
    ffmpeg.output(video, output_path).run()

def process_single_image(image_prediction, filename):
    pose_data = image_prediction.prediction

    with open(f"test/skeletal_data_{filename}.txt", "a") as f:
        f.write(str(pose_data.poses.tolist())+"\n")
        f.write(str(pose_data.bboxes_xyxy.tolist())+"\n")
        f.write("\n")
        f.close()

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

def track_skier(video_name, skier_number, start_frame):
    global skier, previous_pose, previous_box, frames_skipped
    skier = skier_number
    with open(f"test/skeletal_data_{video_name}.txt", "r") as f:
        data = f.read().split("\n\n")
        all_poses = []
        all_boxes = []
        for frame in data:
            if frame == "":
                continue
            poses = eval(frame.split("\n")[0])
            boxes = eval(frame.split("\n")[1])
            all_poses.append(poses)
            all_boxes.append(boxes)
        previous_pose = []
        previous_box = []
        frames_skipped = 0
        with open(f"test/skier_tracked_{video_name}.txt", "w") as dataset_f:
            for frame in range(start_frame, len(all_poses)):
                if all_poses[frame] == []:
                    frames_skipped += 1
                else:
                    skier = skier_num(all_poses[frame], all_boxes[frame])
                dataset_f.write(str(all_poses[frame][skier] if (skier != -1 and skier < len(all_poses[frame])) else []))
                dataset_f.write("\n")
                dataset_f.write(str(all_boxes[frame][skier] if (skier != -1 and skier < len(all_boxes[frame])) else []))
                dataset_f.write("\n\n")
            dataset_f.close()

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

def main():
    # files = glob.glob('test/*')
    # for f in files:
    #     os.remove(f)

    # # Download the video
    video_id = extract_video_id(video_url)
    video_name = f"{video_id}_{start_time}_{end_time}"
    # download_video(video_url, video_id, start_time, end_time, f"test/tmp_{video_name}.mp4")
    # reduce_fps(f"test/tmp_{video_name}.mp4", f"test/{video_name}.mp4", 30.0)
    # os.remove(f"test/tmp_{video_name}.mp4")

    # # Get skeletal data
    # with open(f"test/skeletal_data_{video_name}.txt", "w") as f:
    #     f.write("")
    #     f.close()
    # yolo_nas_pose = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose").cuda()
    # predictions = yolo_nas_pose.to("cuda").predict(f"test/{video_name}.mp4", conf=.1)
    # for image_prediction in predictions._images_prediction_gen:
    #     process_single_image(image_prediction, video_name)

    # # Track skier
    # track_skier(video_name, skier_number, start_frame)

    # Get normalized coordinates
    normalized_coordinates = get_normalized_coordinates(video_name)

    # Load the Keras model
    model = load_model('models/model.keras')

    predictions = [-1] * subframe_length
    for subframe in range(subframe_length, len(normalized_coordinates)):
        subframes = normalized_coordinates[subframe-subframe_length:subframe]
        if [] in subframes:
            predictions.append(-1)
            continue
        subframes = np.array(subframes)
        subframes = subframes.reshape((subframe_length, 34))
        subframes = np.expand_dims(subframes, axis=0)
        prediction = model.predict(subframes, verbose=None)
        predictions.append(float(prediction[0][0]))
    print(predictions)
    
    # Save the predictions to a file
    # with open(f"test/predictions_{video_name}.txt", "w") as f:
    #     for prediction in predictions:
    #         f.write(str(prediction) + "\n")



if __name__ == '__main__':
    main()