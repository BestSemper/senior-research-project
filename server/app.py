from flask import Flask, request, render_template, redirect, url_for
import glob
import re
import os
import torch
from super_gradients.training import models
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
import scripts

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


def extract_video_id(youtube_url):
    # Standard format: https://www.youtube.com/watch?v=VIDEO_ID
    match = re.search(r'v=([A-Za-z0-9_-]+)', youtube_url)
    if match:
        return match.group(1)

    # Shortened format: https://youtu.be/VIDEO_ID
    match = re.search(r'youtu\.be/([A-Za-z0-9_-]+)', youtube_url)
    if match:
        return match.group(1)

    # YouTube shorts format: https://www.youtube.com/shorts/VIDEO_ID
    match = re.search(r'/shorts/([A-Za-z0-9_-]+)', youtube_url)
    if match:
        return match.group(1)

    raise ValueError('Invalid YouTube URL')


@app.route('/download_video', methods=['POST'])
def download_video():
    files = glob.glob('static/videos/*')
    for f in files:
        os.remove(f)
    files = glob.glob('static/files/*')
    for f in files:
        os.remove(f)
    
    video_url = request.form.get('video_url')
    video_id = extract_video_id(video_url)
    start_time = int(request.form.get('start_time'))
    end_time = int(request.form.get('end_time'))
    video_name = f'{video_id}_{start_time}_{end_time}'

    # Download the video
    scripts.download_video(video_url, start_time, end_time, f"static/videos/tmp_{video_name}.mp4")
    scripts.reduce_fps(f"static/videos/tmp_{video_name}.mp4", f"static/videos/{video_name}.mp4", 30.0)
    os.remove(f"static/videos/tmp_{video_name}.mp4")

    return redirect(url_for('downloaded', video_name=video_name))


@app.route('/downloaded')
def downloaded():
    video_name = request.args.get('video_name')
    return render_template('downloaded.html', video_name=video_name)


@app.route('/superimpose', methods=['POST'])
def superimpose():
    video_name = request.form.get('video_name')

    # Get pose data
    with open(f"static/files/pose_data_{video_name}.txt", "w") as f:
        f.write("")
        f.close()
    
    # Make sure that YOLO-NAS-POSE is using the best available hardware
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    yolo_nas_pose = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose").to(device)
    predictions = yolo_nas_pose.predict(f"static/videos/{video_name}.mp4", conf=.1)
    processed_images = [
        scripts.process_single_image(image_prediction, video_name, frame_num)
        for frame_num, image_prediction in enumerate(predictions._images_prediction_gen)
    ]
    if os.path.exists(f'static/videos/superimposed_{video_name}.mp4'):
        os.remove(f'static/videos/superimposed_{video_name}.mp4')
    scripts.create_superimposed_video_from_images(processed_images, f'static/videos/tmp_superimposed_{video_name}.mp4', fps=30.0)
    scripts.encode_video(f'static/videos/tmp_superimposed_{video_name}.mp4', f'static/videos/superimposed_{video_name}.mp4')
    os.remove(f'static/videos/tmp_superimposed_{video_name}.mp4')

    return redirect(url_for('superimposed', video_name=video_name))


@app.route('/superimposed')
def superimposed():
    video_name = request.args.get('video_name')
    return render_template('superimposed.html', video_name=video_name)


@app.route('/track_skier', methods=['POST'])
def track_skier():
    video_name = request.form.get('video_name')
    skier_number = int(request.form.get('skier_number'))
    start_frame = int(request.form.get('start_frame'))

    # Track skier
    scripts.track_skier(video_name, skier_number, start_frame)
    poses, boxes = scripts.get_tracked_skier(video_name)

    if os.path.exists(f"static/videos/skier_tracked_{video_name}.mp4"):
        os.remove(f"static/videos/skier_tracked_{video_name}.mp4")
    scripts.create_tracked_video_from_images(video_name, poses, boxes, f"static/videos/tmp_tracked_{video_name}.mp4", fps=30.0)
    scripts.encode_video(f"static/videos/tmp_tracked_{video_name}.mp4", f"static/videos/tracked_{video_name}.mp4")
    os.remove(f"static/videos/tmp_tracked_{video_name}.mp4")

    return redirect(url_for('skier_tracked', video_name=video_name))


@app.route('/skier_tracked')
def skier_tracked():
    video_name = request.args.get('video_name')
    return render_template('skier_tracked.html', video_name=video_name)


@app.route('/run_model', methods=['POST'])
def run_model():
    video_name = request.form.get('video_name')
    model = request.form.get('model')
    subframe_length = 30

    # Get normalized coordinates
    normalized_coordinates = scripts.get_normalized_coordinates(video_name)

    # Load the Keras model
    model = load_model(f'static/models/{model}.keras', compile=False)

    NOT_FOUND = "skier not found"
    predictions = [NOT_FOUND] * subframe_length
    raw_predictions = [NOT_FOUND] * subframe_length
    for subframe in range(subframe_length, len(normalized_coordinates)):
        subframes = normalized_coordinates[subframe - subframe_length : subframe]
        if [] in subframes:
            predictions.append(NOT_FOUND)
            continue
        subframes = np.array(subframes)
        subframes = subframes.reshape((subframe_length, 34))
        subframes = np.expand_dims(subframes, axis=0)
        prediction = model.predict(subframes, verbose=None)
        raw_predictions.append(prediction[0][0])
        rating = scripts.get_rating(prediction[0][0])
        rating = round(rating, 2)
        predictions.append(rating)
    print(f"Raw predictions: {raw_predictions}")
    print(f"Predictions: {predictions}")

    poses, boxes = scripts.get_tracked_skier(video_name)
    
    # Create video with predictions
    if os.path.exists(f"static/videos/output_{video_name}.mp4"):
        os.remove(f"static/videos/output_{video_name}.mp4")
    scripts.create_final_video_from_images(video_name, predictions, poses, boxes, f"static/videos/tmp_output_{video_name}.mp4", fps=10.0)
    scripts.encode_video(f"static/videos/tmp_output_{video_name}.mp4", f"static/videos/output_{video_name}.mp4")
    os.remove(f"static/videos/tmp_output_{video_name}.mp4")

    return redirect(url_for('results', video_name=video_name))


@app.route('/results')
def results():
    video_name = request.args.get('video_name')
    return render_template('results.html', video_name=video_name)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)