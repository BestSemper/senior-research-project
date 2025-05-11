# Ski Racing Video Analysis

I created an AI model trained to automatically analyze a skier’s form and objectively quantify their skill through video analysis, providing clear, actionable feedback for performance improvement.

![image](images/processed_sample1.png)



# The Process
All coding and model training was done on a Macbook Pro with Apple Silicon.

1. Collect many YouTube slalom ski racing videos and record each skier's slalom points given in the U.S. Ski and Snowboard [website](https://www.usskiandsnowboard.org/public-tools/members).
2. Put all of the data into `training/videos.txt` and download each YouTube video by running `training/download_video.py`.
3. Extract the pose data of the skier in the form of coordinates with YOLO-NAS-POSE by running `training/get_pose_data.py`.
4. Track the skier throughout the video by running `training/track_skier.py`.
5. Train the model using tensorflow by running `training/train_model.py`


# Installation
This project primarily uses Python for all parts, including both frontend and backend. All code was run with [Python 3.9.6](https://www.python.org/downloads/) to avoid package version conflicts. Before running any code, create a virtual environment:

```sh
python -m venv venv
```

Check if you are using the correct Python version by running:

```sh
python --version
```

It should output "Python 3.9.6". Go into the virtual environment with:

```sh
source venv/bin/activate
```

Then, install all required packages by running

```sh
pip install -r requirements.txt
```

If you are using macOS and want to utilize Mac GPUs for model training, also install [tensorflow-metal](https://developer.apple.com/metal/tensorflow-plugin/).


# Model Testing
To test the model, either run `training/test_model.py` or test in the server by running:

```sh
python server/app.py
```


# Issues
If there are issues with loading pre-trained weights for super-gradients, look at this [GitHub Issue](https://github.com/Deci-AI/super-gradients/issues/2064).