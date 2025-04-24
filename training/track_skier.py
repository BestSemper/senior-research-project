import glob
import math
import os


def skier_num(poses, boxes, *args):
    # Return the skier with the least error from the previous frame

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
    skier = skier_number
    previous_pose = []
    previous_box = []
    frames_skipped = 0

    # Read in the pose data
    with open(f"pose_data/{video_name}.txt", "r") as f:
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
            
    with open(f"skier_tracked/{video_name}.txt", "w") as dataset_f:
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


def find_video_index(videos, video_name):
    video_id = video_name[:video_name.rfind('_')]
    start_time = int(video_name[video_name.rfind('_')+1:video_name.rfind('-')])
    end_time = int(video_name[video_name.rfind('-')+1:])
    for i in range(len(videos)):
        if video_id in videos[i][0] and start_time == videos[i][1] and end_time == videos[i][2]:
            return i


def main():
    # Clear the skier_tracked directory
    files = glob.glob('skier_tracked/*')
    for f in files:
        os.remove(f)
    
    # Read in all video URLs
    with open("videos.txt", "r") as f:
        videos = eval(f.read())

    # Process each pose_data file and write the tracked data to skier_tracked
    files = glob.glob("pose_data/*")
    for filename in sorted(files):
        filename = filename.split("pose_data/")[1]
        filename = filename.split(".txt")[0]
        video_index = find_video_index(videos, filename)
        skier_number = videos[video_index][3]
        start_frame = videos[video_index][4]
        print(filename)
        track_skier(filename, skier_number, start_frame)

if __name__ == '__main__':
    main()