# Imports
import numpy as np
import tensorflow as tf
import helpers
import os
import matplotlib.image as mpimg
from PIL import Image
from scipy import misc
tf.logging.set_verbosity(tf.logging.INFO)

TRAINING_SIZE = 20


# Our application logic will be added here

def label_to_img(imgwidth, imgheight, w, h, labels, index=1):
    array_labels = np.zeros([imgwidth, imgheight, 2])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            array_labels[j:j + w, i:i + h] = labels[idx]
            idx = idx + 1
    return array_labels


def value_to_class(v):
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df < foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]


def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j:j + w, i:i + h]
            else:
                im_patch = im[j:j + w, i:i + h, :]
            list_patches.append(im_patch)
    return list_patches


def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""

    def img_to_mask(img):
        if len(img.shape) == 2:
            return img
        return img[:, :, 0]

    gt_imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = img_to_mask(mpimg.imread(image_filename))
            gt_imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    gt_patches = np.asarray([img_crop(gt_imgs[i], 16, 16) for i in range(num_images)])
    data = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = np.asarray([value_to_class(np.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(np.float32)


def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    #N_PATCHES_PER_IMAGE = (IMG_WIDTH / 16) * (IMG_HEIGHT / 16)

    img_patches = [img_crop(imgs[i], 16, 16) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    return np.asarray(data)

def rgb2gray(rgb):
    return np.float32(np.dot(rgb[...,:3], [0.299, 0.587, 0.114]))

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 16, 16, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Output : [batch_size, 16, 16, 1]

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Output : [batch_size, 8, 8, 1]



    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Output : [batch_size, 4, 4, 1]

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Output : [batch_size, 2, 2, 1]



    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 4 * 4 * 64])

    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)

    print("conv1: {}".format(conv1.shape))
    print("conv2: {}".format(conv2.shape))
    print("pool1: {}".format(pool1.shape))
    print("pool2: {}".format(pool2.shape))
    print("dense: {}".format(dense.shape))
    print("dropout: {}".format(dropout.shape))
    print("onehot: {}".format(onehot_labels.shape))

    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Load training and eval data
    data_dir = 'training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/'
    # train_house_labels_filename = data_dir + 'housetruth/'

    # Extract it into numpy arrays.
    tmp_train_data = extract_data(train_data_filename, TRAINING_SIZE)

    tmp_train_data = np.apply_along_axis(rgb2gray, 3, tmp_train_data)
    train_data = np.reshape(tmp_train_data, [-1, 16*16])

    train_labels = extract_labels(train_labels_filename, TRAINING_SIZE)[:,0]

    tmp_eval_data = extract_data(train_data_filename, 1)
    tmp_eval_data = np.apply_along_axis(rgb2gray, 3, tmp_eval_data)
    eval_data = np.reshape(tmp_eval_data, [-1, 16*16])
    eval_labels = np.zeros([625])

    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/model")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=32,
        num_epochs=None,
        shuffle=True)
    classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)

    predictions = list(classifier.predict(input_fn=eval_input_fn))
    predicted_classes = [p["classes"] for p in predictions]

    img = label_to_img(400, 400, 16, 16, predicted_classes) * 255

    img_ = np.zeros([400, 400, 3])
    img_[:, :, 0] = img[:, :, 0]
    img_[:, :, 1] = img[:, :, 1]

    misc.imsave('prediction.png', img_)

    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
