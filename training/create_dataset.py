import glob
import os

import cv2
import ffmpeg
import numpy

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

def process_file(filename):
    with open(f"skier_tracked/{filename}.txt", "r") as f:
        data = f.read().split("\n\n")
        dimensions = eval(data[0])
        all_normalized_coordinates = []
        for frame in data[1:]:
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

def find_video_index(videos, video_name):
    video_id = video_name[:video_name.rfind('_')]
    start_time = int(video_name[video_name.rfind('_')+1:video_name.rfind('-')])
    end_time = int(video_name[video_name.rfind('-')+1:])
    for i in range(len(videos)):
        if video_id in videos[i][0] and start_time==videos[i][1] and end_time==videos[i][2]:
            return i

def get_skeleton_image(pose):
    image = numpy.zeros((1000, 1000, 3), dtype=numpy.uint8)

    if len(pose) == 0:
        return image
    
    EDGE_LINKS = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 6], [5, 7], [5, 11], [6, 8], [6, 12], [7, 9], [8, 10], [11, 12], [11, 13], [12, 14], [13, 15], [14, 16]]
    JOINT_THICKNESS = 5
    
    for i in range(len(EDGE_LINKS)):
        cv2.line(image, (int(pose[EDGE_LINKS[i][0]][0]), int(pose[EDGE_LINKS[i][0]][1])), (int(pose[EDGE_LINKS[i][1]][0]), int(pose[EDGE_LINKS[i][1]][1])), (0, 255, 255), JOINT_THICKNESS)
    
    for i in range(len(pose)):
        cv2.circle(image, (int(pose[i][0]), int(pose[i][1])), 10, (0, 0, 255), -1)

    return image

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
    # files = glob.glob('processed_videos/*')
    # for f in files:
    #     os.remove(f)
    
    with open("videos.txt", "r") as f:
        video_urls = eval(f.read())
    print(video_urls)
    files = glob.glob("skier_tracked/*")
    with open("dataset.txt", "w") as dataset_f:
        dataset_f.write("")
        dataset_f.close()
    for filename in sorted(files):
        filename = filename.split("skier_tracked/")[1]
        filename = filename.split(".txt")[0]
        video_index = find_video_index(video_urls, filename)
        slalom_points = video_urls[video_index][5]
        with open("dataset.txt", "a") as f:
            f.write(filename+"\n")
            f.write(str(slalom_points)+"\n")
            normalized_coordinates = process_file(filename)
            frames = []
            for idx, frame in enumerate(normalized_coordinates):
                if frame == [] and idx < len(normalized_coordinates)-1 and normalized_coordinates[idx+1] != []:
                    f.write("\n\n")
                    f.write(filename+"\n")
                    f.write(str(slalom_points)+"\n")
                elif frame != []:
                    f.write(str(frame))
                    f.write("\n")
            f.write("\n\n")
            #     skeleton_image = get_skeleton_image(frame)
            #     frames.append(skeleton_image)
            # create_video_from_frames(frames, f"processed_videos/tmp_{filename}.mp4")
            # encode_video(f"processed_videos/tmp_{filename}.mp4", f"processed_videos/{filename}.mp4")
            # os.remove(f"processed_videos/tmp_{filename}.mp4")

if __name__ == '__main__':
    main()