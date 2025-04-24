# senior-research-project

I created an AI model trained to automatically analyze a skier’s form and objectively quantify their skill through video analysis, providing clear, actionable feedback for performance improvement.

# Local Testing
- Model training was done on a Macbook Pro M4 Max 


# Install
This project primarily uses Python for all parts, including both frontend and backend. All code was run with Ptyhon 3.9.6 to avoid package version conflicts. For testing the code, first download [Python 3.9.6](https://www.python.org/downloads/) if not installed already. Then, create a virtual environment:

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

If you are using a MacOS and want to utilize Mac GPUs for model training, also install [tensorflow-metal](https://developer.apple.com/metal/tensorflow-plugin/).