# -*- coding:utf-8 -*-

import tensorflow as tf
from PIL import Image
import numpy as np
import PIL.ImageOps


def convert_image(image):
    image = image.resize((28, 28), Image.ANTIALIAS)
    image = image.convert('L')
    # image.show()
    image = PIL.ImageOps.invert(image)
    image = np.array(image).astype(np.float32)
    image = image.reshape(-1, height * width)
    image = normalize(image)
    return image


def normalize(x):
    return x / 255


def neural_net_image_input(image_shape):
    return tf.placeholder(tf.float32, (None, image_shape[0], image_shape[1], image_shape[2]), name='x')


def neural_net_label_input(n_classes):
    return tf.placeholder(tf.float32, (None, n_classes), name='y')


def neural_net_keep_prob_input():
    return tf.placeholder(tf.float32, name='keep_prob')


def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    weight = tf.Variable(
        tf.truncated_normal((list(conv_ksize) + [x_tensor.get_shape().as_list()[3], conv_num_outputs]), stddev=0.04))
    bias = tf.Variable(tf.zeros(conv_num_outputs))
    # 2D Convolution Layer
    output = tf.nn.conv2d(x_tensor,
                          weight,
                          [1, conv_strides[0], conv_strides[1], 1],
                          padding='SAME')
    output = tf.nn.bias_add(output, bias)
    # Pooling Layer
    output = tf.nn.max_pool(output,
                            [1, pool_ksize[0], pool_ksize[1], 1],
                            [1, pool_strides[0], pool_strides[1], 1],
                            padding='SAME')
    return output


def flatten(x_tensor):
    shape = x_tensor.get_shape().as_list()
    return tf.reshape(x_tensor, [-1, shape[1] * shape[2] * shape[3]])


def fully_conn(x_tensor, num_outputs):
    weight = tf.Variable(tf.truncated_normal((x_tensor.get_shape().as_list()[1], num_outputs), stddev=0.04))
    bias = tf.Variable(tf.zeros(num_outputs))
    output = tf.add(tf.matmul(x_tensor, weight), bias)
    return tf.nn.relu(output)


def output(x_tensor, num_outputs):
    weight = tf.Variable(tf.truncated_normal((x_tensor.get_shape().as_list()[1], num_outputs), stddev=0.04))
    bias = tf.Variable(tf.zeros(num_outputs))
    output = tf.add(tf.matmul(x_tensor, weight), bias)
    return output


def conv_net(x, keep_prob):
    conv = conv2d_maxpool(x,
                           conv_num_outputs=64,
                           conv_ksize=[3,3],
                           conv_strides=[1,1],
                           pool_ksize=[2,2],
                           pool_strides=[2,2])

    conv = conv2d_maxpool(conv,
                          conv_num_outputs=128,
                          conv_ksize=[3,3],
                          conv_strides=[1,1],
                          pool_ksize=[2,2],
                          pool_strides=[2,2])

    conv = conv2d_maxpool(conv,
                          conv_num_outputs=256,
                          conv_ksize=[3,3],
                          conv_strides=[1,1],
                          pool_ksize=[2,2],
                          pool_strides=[2,2])

    conv = conv2d_maxpool(conv,
                          conv_num_outputs=512,
                          conv_ksize=[3,3],
                          conv_strides=[1,1],
                          pool_ksize=[2,2],
                          pool_strides=[2,2])

    conv = conv2d_maxpool(conv,
                          conv_num_outputs=1024,
                          conv_ksize=[3,3],
                          conv_strides=[1,1],
                          pool_ksize=[2,2],
                          pool_strides=[2,2])

    flt = flatten(conv)

    fc = fully_conn(flt, 512)
    fc = fully_conn(fc, 512)
    fc = fully_conn(fc, 1024)
    fc = tf.nn.dropout(fc, keep_prob)

    o = output(fc, len(image_names))
    return o

# def index2word(image_names):
#     arr = {}
#     for index, name in enumerate(image_names):
#         arr[index] = name
#     return arr


image_name_file = '/Volumes/INFO/xxdr/Documents/Python/quick_draw/mini_quick_draw.txt'
image_names = []
with open(image_name_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if lines:
            image_names.append(line.replace('\n', ''))
# index2wordDic = index2word(image_names)

width, height, n_len, n_class = 28, 28, 1, len(image_names)
x = neural_net_image_input((height, width, 1))
keep_prob = neural_net_keep_prob_input()

logits = conv_net(x, keep_prob)
correct_pred = tf.argmax(logits, 1)

save_model_path = '/Volumes/INFO/xxdr/Documents/Python/quick_draw/image_classification'
saver = tf.train.Saver()

def scaner(image):
    features = np.zeros((1, height * width))
    features[0, :] = convert_image(image)
    features = features.reshape(-1, height, width, 1)
    with tf.Session() as sess:
        saver.restore(sess, save_model_path)
        # print(sess.run(logits, feed_dict={x: features, keep_prob: 1.}))
        return image_names[sess.run(correct_pred, feed_dict={x: features, keep_prob: 1.})[0]]
