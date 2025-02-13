import glob
import math
import os

skier = 0
previous_pose = []
previous_box = []
frames_skipped = 0

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

def track_single_image(poses, boxes):
    global skier, frames_skipped

    if poses == []:
        frames_skipped += 1
        return

    skier = skier_num(poses, boxes)

def track_skier(video_name, skier_number, start_frame):
    global skier, previous_pose, previous_box, frames_skipped
    skier = skier_number
    with open(f"skeletal_data/{video_name}.txt", "r") as f:
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
        with open(f"skier_tracked/{video_name}.txt", "w") as dataset_f:
            dataset_f.write(str(dimensions)+"\n\n")
            for frame in range(start_frame, len(all_poses)):
                track_single_image(all_poses[frame], all_boxes[frame])
                dataset_f.write(str(all_poses[frame][skier] if (skier != -1 and skier < len(all_poses[frame])) else []))
                dataset_f.write("\n")
                dataset_f.write(str(all_boxes[frame][skier] if (skier != -1 and skier < len(all_boxes[frame])) else []))
                dataset_f.write("\n\n")
            dataset_f.close()

def find_video_index(videos, video_name):
    video_id = video_name[:video_name.rfind('_')]
    start_time = int(video_name[video_name.rfind('_')+1:video_name.rfind('-')])
    end_time = int(video_name[video_name.rfind('-')+1:])
    for i in range(len(videos)):
        if video_id in videos[i][0] and start_time==videos[i][1] and end_time==videos[i][2]:
            return i

def main():
    files = glob.glob('skier_tracked/*')
    for f in files:
        os.remove(f)
    
    with open("videos.txt", "r") as f:
        videos = eval(f.read())
    print(videos)

    files = glob.glob("skeletal_data/*")
    for filename in sorted(files):
        filename = filename.split("skeletal_data/")[1]
        filename = filename.split(".txt")[0]
        video_index = find_video_index(videos, filename)
        skier_number = videos[video_index][3]
        start_frame = videos[video_index][4]
        print(filename)
        track_skier(filename, skier_number, start_frame)

if __name__ == '__main__':
    main()