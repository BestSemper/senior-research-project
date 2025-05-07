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
    files = glob.glob('static/videos/*')
    for f in files:
        os.remove(f)
    files = glob.glob('static/files/*')
    for f in files:
        os.remove(f)
    
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
    video_url = request.form.get('video_url')
    video_id = extract_video_id(video_url)
    start_time = int(request.form.get('start_time'))
    end_time = int(request.form.get('end_time'))
    video_name = f'{video_id}_{start_time}_{end_time}'

    return redirect(url_for('downloaded', video_name=video_name))


@app.route('/downloaded')
def downloaded():
    video_name = request.args.get('video_name')
    return render_template('downloaded.html', video_name=video_name)


@app.route('/superimpose', methods=['POST'])
def superimpose():
    video_name = request.form.get('video_name')
    return redirect(url_for('superimposed', video_name=video_name))


@app.route('/superimposed')
def superimposed():
    video_name = request.args.get('video_name')
    return render_template('superimposed.html', video_name=video_name)


@app.route('/track_skier', methods=['POST'])
def track_skier():
    video_name = request.form.get('video_name')
    return redirect(url_for('skier_tracked', video_name=video_name))


@app.route('/skier_tracked')
def skier_tracked():
    video_name = request.args.get('video_name')
    return render_template('skier_tracked.html', video_name=video_name)


@app.route('/run_model', methods=['POST'])
def run_model():
    video_name = request.form.get('video_name')
    return redirect(url_for('results', video_name=video_name))


@app.route('/results')
def results():
    video_name = request.form.get('video_name')
    return render_template('results.html', video_name=video_name)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)