import glob
import math

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

def track_single_image(dimensions, poses, boxes):
    global frames_skipped

    if poses == []:
        frames_skipped += 1
        return

    skier_num(poses, boxes)

def track_skier(video_name, skier_number, start_frame):
    global skier, previous_pose, previous_box, frames_skipped
    skier = skier_number
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
        skier_tracked = []
        for frame in range(start_frame, len(all_poses)):
            track_single_image(dimensions, all_poses[frame], all_boxes[frame])
            skier_pose = poses[skier] if skier != -1 and (skier < len(poses)) else []
            skier_tracked.append(skier_pose)
        with open(f"static/output/tracked_{video_name}.txt", "w") as dataset_f:
            dataset_f.write(str([int(dimensions[0]), int(dimensions[1])])+"\n")
            for frame in skier_tracked:
                dataset_f.write(str(frame)+"\n")
            dataset_f.close()

def main():
    with open("videos.txt", "r") as f:
        video_urls = eval(f.read())
    print(video_urls)
    files = glob.glob("skeletal_data/*")
    for filename in sorted(files):
        filename = filename.split("output/")[1]

if __name__ == '__main__':
    main()