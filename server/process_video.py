import cv2
from super_gradients.training import models
from super_gradients.training.utils.visualization.pose_estimation import PoseVisualization
import numpy
import os

def process_single_image(image_prediction):
    image = image_prediction.image
    pose_data = image_prediction.prediction

    with open("output/data.txt", "a") as f:
        f.write(str(pose_data.poses)+"\n\n")
    
    blank_image = numpy.zeros_like(image)  # for a black background

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

def create_video_from_frames(frames, output_filename="output_video.mp4", fps=30.0):
    """
    Create an mp4 video from a list of image frames.

    Parameters:
    - frames : list of np.ndarray
        List of image frames represented as numpy arrays.
    - output_filename : str, optional
        Name of the output video file.
    - fps : float, optional
        Frames per second for the output video.

    Returns:
    - None
    """

    # Determine the width and height from the first frame
    frame_height, frame_width, layers = frames[0].shape

    # Define the codec for .mp4 format
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    # Write each frame to the video
    for frame in frames:
        out.write(frame)

    # Close and release everything
    out.release()
    cv2.destroyAllWindows()

def main():
    with open("output/data.txt", "w") as f:
        f.write("")
    yolo_nas_pose = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose").cuda()

    for filename in next(os.walk("./videos"), (None, None, []))[2] :
        predictions = yolo_nas_pose.to("cuda").predict(f"videos/{filename}", conf=.4)
        processed_frames = [process_single_image(image_prediction) for image_prediction in predictions._images_prediction_gen]
        #create_video_from_frames(processed_frames, f"output/processed_{filename}", fps=30.0)
        break

if __name__ == "__main__":
    main()