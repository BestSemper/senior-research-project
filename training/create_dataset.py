

import glob


with open("videos.txt", "r") as f:
    video_urls = eval(f.read())
print(video_urls)

def main():
    files = glob.glob("output/*")
    for filename in sorted(files):
        filename = filename.split("output/")[1]
    

if __name__ == '__main__':
    main()