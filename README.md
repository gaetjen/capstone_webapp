# Car damage assessment webapp

This webapp provides a user-friendly interface for the classifiers and visualization tool developed for the car damage assessment capstone project from VKB during the data incubator.

## Instructions
### Requirements
* Python 3
* pip
* Git (available from the command line)
### Installation
#### On Linux:
```
git clone https://github.com/gaetjen/capstone_webapp.git
cd capstone_webapp
python3 -m venv app_env
. app_env/bin/activate
pip install -r requirements.txt
export FLASK_APP=app.py
```
#### On Windows:
```
git clone https://github.com/gaetjen/capstone_webapp.git
cd capstone_webapp
py -3 -m venv app_env
venv\Scripts\activate
pip install -r requirements.txt
set FLASK_APP=app.py
```
TODO: downloading models
#### Optional
If you have a system set up to use tensorflow on the gpu the following may lead to speedups in classification:
```
pip uninstall tensorflow
pip install tensorflow-gpu
```
### Running
You can use `flask run` to start the app and access it at the web address http://127.0.0.1:5000/. Alternatively you can use `python app.py` and the site will be available from your own computer at http://0.0.0.0:33507/ and also from other computers via your public IP address.

On the main site there is a form to upload images. After upload and classification is complete the images are shown together with the prediction and confidence, as well as a visualization of the image regions relevant for the damage/no-damage classifier.

## Limitations
The app is not set up to work properly with multiple users in parallel.