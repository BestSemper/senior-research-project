import cv2
from super_gradients.training import models
from super_gradients.training.utils.visualization.pose_estimation import PoseVisualization
import os
import glob
import ffmpeg
import torch


def process_single_image(image_prediction, video_name):
    """
    Process a single image and save the pose data.

    Returns the image with pose data drawn on it.
    """
    pose_data = image_prediction.prediction
    image = image_prediction.image

    # Record the pose data for each frame
    with open(f"pose_data/{video_name}.txt", "a") as f:
        f.write(str(pose_data.poses.tolist()) + "\n")
        f.write(str(pose_data.bboxes_xyxy.tolist()) + "\n")
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

    return skeleton_image


def create_video_from_images(images, output_video_name="output_video.mp4", fps=30.0):
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


def encode_video(input_path, output_path):
    """
    Make sure the video is in the correct format
    """
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
    print(f"Using device: {device}")
    yolo_nas_pose = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose").to(device)

    # Process each video and write the pose data to pose_data
    videos = glob.glob("videos/*")
    for video_name in sorted(videos):
        video_name = video_name.split("videos/")[1]
        video_name = video_name.split(".mp4")[0]

        print(f"Processing video: {video_name}")
        predictions = yolo_nas_pose.predict(f"videos/{video_name}.mp4", conf=0.1)
        processed_images = [
            process_single_image(image_prediction, video_name)
            for image_prediction in predictions._images_prediction_gen
        ]

        # Uncomment the following lines to create a video showing the pose data
        # create_video_from_images(processed_images, f'output/tmp_{video_name}.mp4', fps=30.0)
        # encode_video(f'output/tmp_{video_name}.mp4', f'output/{video_name}.mp4')
        # os.remove(f'output/tmp_{video_name}.mp4')


if __name__ == "__main__":
    main()
