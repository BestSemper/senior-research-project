import cv2
from super_gradients.training import models
from super_gradients.training.utils.visualization.pose_estimation import (
    PoseVisualization,
)
import os
import glob
import ffmpeg
import torch


def process_single_image(image_prediction, filename):
    pose_data = image_prediction.prediction

    # Record the pose data for each frame
    with open(f"pose_data/{filename}.txt", "a") as f:
        f.write(str(pose_data.poses.tolist()) + "\n")
        f.write(str(pose_data.bboxes_xyxy.tolist()) + "\n")
        f.write("\n")
        f.close()


def create_video_from_frames(frames, output_filename="output_video.mp4", fps=30.0):
    frame_height, frame_width, layers = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    for frame in frames:
        video.write(frame)

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
    # Clear the pose_data directory
    files = glob.glob("pose_data/*")
    for f in files:
        os.remove(f)

    # Make sure that YOLO-NAS-POSE is using the best available hardware
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    yolo_nas_pose = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose").to(
        device
    )

    # Process each video and write the pose data to pose_data
    files = glob.glob("videos/*")
    for filename in sorted(files):
        filename = filename.split("videos/")[1]
        filename = filename.split(".mp4")[0]

        print(f"Processing video: {filename}")
        predictions = yolo_nas_pose.predict(f"videos/{filename}.mp4", conf=0.1)
        processed_frames = [
            process_single_image(image_prediction, filename)
            for image_prediction in predictions._images_prediction_gen
        ]

        # Uncomment the following lines to create a video showing the pose data
        # create_video_from_frames(processed_frames, f'output/tmp_{filename}.mp4', fps=30.0)
        # encode_video(f'output/tmp_{filename}.mp4', f'output/{filename}.mp4')
        # os.remove(f'output/tmp_{filename}.mp4')


if __name__ == "__main__":
    main()
