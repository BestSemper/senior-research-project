import cv2
from super_gradients.training import models
from super_gradients.training.utils.visualization.pose_estimation import PoseVisualization
import numpy
import os
import glob
import ffmpeg

def process_single_image(image_prediction, filename):
    image = image_prediction.image
    pose_data = image_prediction.prediction

    with open(f"output/{filename}.txt", "a") as f:
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
    return skeleton_image

def create_video_from_frames(frames, output_filename='output_video.mp4', fps=30.0):
    frame_height, frame_width, layers = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    for frame in frames:
        video.write(frame)

    video.release()
    cv2.destroyAllWindows()

def encode_video(input_path, output_path):
    ffmpeg.input(input_path).output(output_path, vcodec='libx264', crf=18, preset='fast', acodec='aac', audio_bitrate='128k').run()

def main():
    # Remove videos from previous sessions to prevent caching errors
    files = glob.glob('output/*')
    for f in files:
        os.remove(f)
    
    yolo_nas_pose = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose").cuda()

    files = glob.glob("videos/*")
    for filename in sorted(files):
        filename = filename.split("videos/")[1]
        filename = filename.split(".mp4")[0]
        video = cv2.VideoCapture(f"videos/{filename}.mp4")
        with open(f"output/{filename}.txt", "w") as f:
            f.write(str([int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))])+"\n\n")
            f.close()
        predictions = yolo_nas_pose.to("cuda").predict(f"videos/{filename}.mp4", conf=.1)
        processed_frames = [process_single_image(image_prediction, filename) for image_prediction in predictions._images_prediction_gen]
        create_video_from_frames(processed_frames, f'output/tmp_{filename}.mp4', fps=30.0)
        encode_video(f'output/tmp_{filename}.mp4', f'output/{filename}.mp4')
        os.remove(f'output/tmp_{filename}.mp4')

if __name__ == "__main__":
    main()