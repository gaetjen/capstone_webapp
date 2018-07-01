#####################################
# set FLASK_APP=app.py              #
# # zum Debuggen: set FLASK_DEBUG=1 #
# flask run                         #
#####################################

from flask import Flask, render_template, request
import os
import dill
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import tensorflow as tf
#import boto3
#from io import BytesIO
import random


# s3 = boto3.resource('s3')

global feature_extractor
global graph
# with open('models/vgg16_notop.h5', 'wb') as data:
#     s3.Bucket("cd-models").download_fileobj("vgg16_notop.h5", data)

feature_extractor = load_model('models/vgg16_notop.h5')
graph = tf.get_default_graph()

# with BytesIO() as data:
#     s3.Bucket("cd-models").download_fileobj("svc_no_pca.pk", data)
#     data.seek(0)    # move back to the beginning after writing
#     classifier = dill.load(data)
classifier = dill.load(open('models/pca_svc.pk', 'rb'))

app = Flask(__name__)
app.config['UPLOADS'] = os.path.join('static/uploads/')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        for f in os.listdir("./static/uploads"):
            os.remove("./static/uploads/" + f)

        f_list = request.files.getlist('filename[]')
        if len(f_list) == 8:
            full_path = []
            result = []
            for idx, f in enumerate(f_list):
                fn = 'uploadfile{}.jpeg'.format(random.randint(0, 99999999))
                full_path1 = os.path.join(app.config['UPLOADS'], fn)
                full_path.append(full_path1)
                f.save(full_path1)
                result.append(get_classification(full_path1))
            return render_template('show_results8.html', im_url=full_path, classification=result)

        else:
            for idx, f in enumerate(f_list):
                fn = 'uploadfile{}.jpeg'.format(random.randint(0, 99999999))
                full_path = os.path.join(app.config['UPLOADS'], fn)
                f.save(full_path)

            result = get_classification(full_path)

            return render_template('show_results.html', im_url=full_path, classification=result)
    if request.method == 'GET':
        return render_template('upload.html')


CLASSES_3 = ["Nahaufnahme", "Außenaufnahme", "Innenaufnahme"]


def get_classification(file_path):
    # TODO: batch processing
    # TODO: set evaluation
    with graph.as_default():
        # print(file_path)
        prepared = prepare_image(file_path)
        # print(prepared.shape)
        vgg_features = feature_extractor.predict(prepared)
        vgg_features = np.reshape(vgg_features, (1, 7 * 7 * 512))
        result = classifier.predict_proba(vgg_features)
        highest = np.argmax(result)
        print(result)
        return CLASSES_3[highest] + ": %2.0f" % (result[0, highest] * 100) + '%'


def prepare_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    return prepare_image_direct(img)


def prepare_image_direct(img):
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x / 255


if __name__ == '__main__':
    app.run(port=33507, host='0.0.0.0')