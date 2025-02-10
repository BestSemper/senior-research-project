import glob
import math
import cv2
import os
import ffmpeg
from super_gradients.training.utils.visualization.pose_estimation import PoseVisualization
import matplotlib.pyplot as plt
import numpy

skier = 0
previous_pose = []
previous_box = []
frames_skipped = 0 

#aTYt, bSXy, PRscy

def track_skier(poses, boxes):
    global skier, previous_pose, previous_box, frames_skipped
    if previous_pose == []:
        previous_pose = poses[skier]
        previous_box = boxes[skier]
        return
    errors = []
    for pose in poses:
        error = 0
        for i in range(17):
            error += math.sqrt((pose[i][0] - previous_pose[i][0])**2 + (pose[i][1] - previous_pose[i][1])**2)
        errors.append(error)
    skier = errors.index(min(errors))
    box_size = max(previous_box[2] - previous_box[0], previous_box[3] - previous_box[1])
    if errors[skier]/17 > box_size/10*(frames_skipped+1):
        skier = -1
        frames_skipped += 1
        return
    frames_skipped = 0
    previous_pose = poses[skier]
    previous_box = boxes[skier]

def process_single_image(dimensions, poses, boxes):
    global frames_skipped
    skeleton_image = numpy.zeros((dimensions[0], dimensions[1], 3), dtype=numpy.uint8)

    if poses == []:
        frames_skipped += 1
        return skeleton_image

    track_skier(poses, boxes)

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
            confidence = (pose[start][2] + pose[end][2]) / 2
            cv2.line(skeleton_image, (x_coords[0], y_coords[0]), (x_coords[1], y_coords[1]), EDGE_COLORS[idx], thickness=2)

    # keypoints
    for idx, keypoint in enumerate(pose):
        cv2.circle(skeleton_image, (int(keypoint[0]), int(keypoint[1])), radius=5, color=KEYPOINT_COLORS[idx], thickness=-1)

    # bounding box
    if len(box) == 4:
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(skeleton_image, (x_min, y_min), (x_max, y_max), BOX_COLOR, thickness=2)

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
    global skier
    # Remove videos from previous sessions to prevent caching errors
    files = glob.glob('tracked_output/*')
    for f in files:
        os.remove(f)
    
    files = glob.glob("output/*")
    for filename in sorted(files):
        filename = filename.split("output/")[1]
        if filename[-3:] != "txt":
            continue
        with open(f"output/{filename}", "r") as f:
            filename = filename[:-4]
            line = f.readline().split(" ")
            skier = int(line[0])
            start_frame = int(line[1])
            f.readline()
            data = f.read().split("\n\n")
            dimensions = eval(data[0])
            processed_frames = []
            skier_tracked = []
            for frame in data[1+start_frame:]:
                if frame == "":
                    continue
                poses = eval(frame.split("\n")[0])
                boxes = eval(frame.split("\n")[1])
                processed_frames.append(process_single_image(dimensions, poses, boxes))
                skier_pose = poses[skier] if skier != -1 and (skier < len(poses)) else []
                skier_tracked.append(skier_pose)
            with open(f"tracked_output/{filename}.txt", "w") as dataset_f:
                dataset_f.write(str([int(dimensions[0]), int(dimensions[1])])+"\n")
                for frame in skier_tracked:
                    dataset_f.write(str(frame)+"\n")
                dataset_f.close()
            create_video_from_frames(processed_frames, f"tracked_output/tmp_{filename}.mp4")
            encode_video(f"tracked_output/tmp_{filename}.mp4", f"tracked_output/{filename}.mp4")
            os.remove(f"tracked_output/tmp_{filename}.mp4")
            f.close()

if __name__ == '__main__':
    main()