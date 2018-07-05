#####################################
# set FLASK_APP=app.py              #
# # zum Debuggen: set FLASK_DEBUG=1 #
# flask run                         #
#####################################

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from flask import Flask, render_template, request
import os
import dill
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg16, xception
import numpy as np
import tensorflow as tf
from collections import Counter
from sympy.utilities.iterables import multiset_permutations
from vis.visualization import visualize_cam
from PIL import Image
#import boto3
#from io import BytesIO
import random


CLASSES_3 = ["Nahaufnahme", "Außenaufnahme", "Innenaufnahme"]
CLASSES_DMG = ["Unbeschädigt", "Beschädigt"]

# s3 = boto3.resource('s3')

# global feature_extractor
# global graph
# with open('models/vgg16_notop.h5', 'wb') as data:
#     s3.Bucket("cd-models").download_fileobj("vgg16_notop.h5", data)

dmg_classifier = load_model('models/classifier-damaged-xception.h5')
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

        saved_files = save_uploads(f_list)
        if len(f_list) <= 8:
            im_ws = get_image_widths(saved_files, 230)
            three_class_results = get_classification(saved_files)
            dmg_results, grad_files = get_dmg_classification(saved_files)
            results = [x + '\n' + y for x, y in zip(three_class_results, dmg_results)]
            return render_template('show_results.html',
                                   im_url=saved_files,
                                   classification=results,
                                   grad_url=grad_files,
                                   im_ws=im_ws)
        else:
            return "Maximal 8 Bilder!"

    if request.method == 'GET':
        return render_template('upload.html')


def get_image_widths(f_list, to_height):
    rtn = []
    for f in f_list:
        with Image.open(f) as img:
            w, h = img.size
            rtn.append(w / h * 230)
    return rtn


def save_uploads(f_list):
    saved_list = []
    for idx, f in enumerate(f_list):
        fn = 'uploadfile{}.jpeg'.format(idx)
        full_path = os.path.join(app.config['UPLOADS'], fn)
        f.save(full_path)
        saved_list.append(full_path)
    return saved_list


def get_dmg_classification(file_paths):
    with graph.as_default():
        im_list = []
        for file_path in file_paths:
            prepared = prepare_image(file_path, target_size=(299, 299), net='xception')
            im_list.append(prepared)
        im_list = np.vstack(im_list)
        predictions = dmg_classifier.predict(im_list)
        grad_files = []
        for idx, im, pr in zip(range(len(im_list)), im_list, predictions):
            grad_fig = make_gradcam(im, pr, dmg_classifier)
            fn = 'gradfile{}.png'.format(idx)
            full_path = os.path.join(app.config['UPLOADS'], fn)
            grad_fig.savefig(full_path, bbox_inches='tight', pad_inches=0)
            grad_files.append(full_path)
        dmg_class = [1 if p[0] > 0.5 else 0 for p in predictions]
        results = [CLASSES_DMG[dc] + ": %2.0f" % (abs(1 - dc - p[0]) * 100) + '%' for dc, p in zip(dmg_class, predictions)]
        return results, grad_files


def get_classification(file_paths):
    with graph.as_default():
        im_list = []
        for file_path in file_paths:
            prepared = prepare_image(file_path)
            im_list.append(prepared)
        vgg_features = feature_extractor.predict(np.vstack(im_list))
        vgg_features = np.reshape(vgg_features, (len(im_list), 7 * 7 * 512))
        predictions = classifier.predict_proba(vgg_features)
        if len(file_paths) == 8:
            highest, _ = predict_set(predictions, [2, 4, 2])
        else:
            highest = np.argmax(predictions, axis=1)
        results = [CLASSES_3[h] + ": %2.0f" % (p[h] * 100) + '%' for p, h in zip(predictions, highest)]
        return results


# TODO: (optional, only for speed) first check if taking max confidence for each sample already gives good distribution
def predict_set(prob_mtx, n_per_cat, labels=None):
    """
    Predict the classes of a set of samples using knowledge about the number of samples belonging to each class
    :param prob_mtx: matrix specifying all class probabilities for every sample
    :param n_per_cat: list of number of samples belonging to each class
    :param labels: optional: give label strings
    :return: class predictions
    """
    if not labels:
        labels = [i for i in range(len(n_per_cat))]
    labels = np.array(labels)
    if prob_mtx.shape[0] != sum(n_per_cat):
        print("total number of assigneld labels must match number of elements")
        return
    if prob_mtx.shape[1] != len(n_per_cat):
        print("probabilities must match the number of categories!")
        return
    label_counts = Counter({l_idx: n for l_idx, n in enumerate(n_per_cat)})
    best_so_far = -np.inf
    best_perm = []
    # go over all possible permutations. multiset_permutations skips the duplicates due to repeated occurrences
    for permutation in multiset_permutations(list(label_counts.elements())):
        current_sum = sum(prob_mtx[np.arange(len(prob_mtx)), permutation])
        if current_sum > best_so_far:
            best_so_far = current_sum
            best_perm = permutation

    return best_perm, labels[best_perm]


def prepare_image(img_path, target_size=(224, 224), net='vgg16'):
    img = load_img(img_path, target_size=target_size)
    return prepare_image_direct(img, net)


def prepare_image_direct(img, net):
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    if net == 'vgg16':
        # x = vgg16.preprocess_input(x)
        # apparently just rescaling works much better than using the vgg16 preprocessing
        x /= 255
    elif net == 'xception':
        x = xception.preprocess_input(x)
    else:
        raise ValueError("Unknown net architecture, don't know how to preprocess image!")
    return x


def normalize_tf_image(image):  # from [-1. ... 1.] to [0. ... 1.]
    return (image + 1.) / 2.


def make_gradcam(image, prediction, model):
    prediction = prediction.squeeze()
    if prediction >= 0.5:
        # highlight the regions contributing towards '1-damaged' classification
        grad_modifier = None
    else:
        # highlight the regions contributing towards '0-non-damaged' classification
        grad_modifier = 'negate'

    # squeeze out the 'batch' dimension for keras-vis
    image = image.squeeze()

    gradient = visualize_cam(model, layer_idx=-1, filter_indices=0,
                             seed_input=image, backprop_modifier=None, grad_modifier=grad_modifier)

    image = normalize_tf_image(image)
    gradient_tmp = np.expand_dims(gradient, axis=2)
    # print(image.shape, gradient_tmp.shape)
    image_gradient = np.concatenate((image, gradient_tmp), axis=2)
    # print(image_gradient.shape)
    figure, ax = plt.subplots(1, 1)
    # figure.suptitle('Prediction: {:.2f}'.format(prediction), y=0.8, horizontalalignment='center')
    ax.imshow(image_gradient)
    ax.contour(gradient, 3, colors='k')
    plt.axis('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return figure


# adapted from stackoverflow: https://stackoverflow.com/questions/34066804/disabling-caching-in-flask
@app.after_request
def add_header(r):
    """
    Add headers to suppress caching.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r


if __name__ == '__main__':
    app.run(port=33507, host='0.0.0.0')