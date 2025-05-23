import yt_dlp
from yt_dlp.utils import download_range_func
import os
import re
import ffmpeg
import glob

cur_dir = os.getcwd()


def extract_video_id(youtube_url):
    # Standard format: https://www.youtube.com/watch?v=VIDEO_ID
    match = re.search(r"v=([A-Za-z0-9_-]+)", youtube_url)
    if match:
        return match.group(1)

    # Shortened format: https://youtu.be/VIDEO_ID
    match = re.search(r"youtu\.be/([A-Za-z0-9_-]+)", youtube_url)
    if match:
        return match.group(1)

    # YouTube shorts format: https://www.youtube.com/shorts/VIDEO_ID
    match = re.search(r"/shorts/([A-Za-z0-9_-]+)", youtube_url)
    if match:
        return match.group(1)

    raise ValueError("Invalid YouTube URL")


def download_video(link, start_time, end_time, download_path):
    """
    Download a video with the best video quality from YouTube using yt-dlp.
    """
    cur_dir = os.getcwd()
    youtube_dl_options = {
        "download_ranges": download_range_func(None, [(start_time, end_time)]),
        "force_keyframes_at_cuts": True,
        "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "format_sort": ["res:1080", "ext:mp4:m4a"],
        "encoding": "utf-8",
        "outtmpl": os.path.join(cur_dir, download_path),
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(youtube_dl_options) as ydl:
            ydl.download([link])
    except Exception as e:
        print(f"Error downloading video: {str(e)}")


def reduce_fps(video_path, output_path, new_fps=30):
    """
    Reduce the FPS of the video to 30 for consistency
    """
    video = ffmpeg.input(video_path)
    video = video.filter("fps", fps=new_fps, round="up")
    ffmpeg.output(video, output_path).run()


def main():
    # Clear the videos directory
    files = glob.glob("videos/*")
    for f in files:
        os.remove(f)

    # Read in all video URLs
    video_urls = []
    with open("videos.txt", "r") as f:
        video_urls = eval(f.read())

    # Download and process each video
    for url in video_urls:
        link = url[0]
        video_id = extract_video_id(url[0])
        start_time = url[1]
        end_time = url[2]
        video_name = f"{video_id}_{start_time}-{end_time}.mp4"
        tmp_video_name = f"{video_id}_{start_time}-{end_time}_tmp.mp4"

        download_video(link, start_time, end_time, f"videos/{tmp_video_name}")
        reduce_fps(f"videos/{tmp_video_name}", f"videos/{video_name}")
        os.remove(f"videos/{tmp_video_name}")


if __name__ == "__main__":
    main()