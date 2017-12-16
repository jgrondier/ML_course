import numpy as np
import tensorflow as tf
import tqdm
import vgg16

with tf.device('/cpu:0'):
    with tf.Session() as sess:
        vgg = vgg16.Vgg16()
        