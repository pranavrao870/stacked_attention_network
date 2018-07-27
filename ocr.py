import tensorflow as tf
import cv2
import numpy as np
import os

import sys

import east.model as model
from east.detect_boxes import detect, resize_image, sort_poly
from crnn.crnn_model.crnn_model import ShadowNet
from crnn.local_utils import data_utils

"""
This will preprocess the whole image into some database
"""

def func(ans, indices):
    for i in indices:
        if not i in ans:
            return i

def sort_by_pos(image_list, boxes,centers, images_shape):
    rows, cols, chan = images_shape
    centers_adjusted = [(rows - y, x) for (y, x) in centers]
    dists = [y**2 + x**2 for (y, x) in centers_adjusted]
    indices = np.argsort(dists)
    ans = []
    num_images = len(image_list)
    ans.append(indices[0])
    while len(ans) < num_images:
        recent = indices[0]
        centers_new = [(centers_adjusted[recent][0] - y, centers_adjusted[recent][1] - x) for (y, x) in centers_adjusted]
        dists = [y**2 + x**2 for (y, x) in centers_new]
        indices = np.argsort(dists)
        ans.append(func(ans, indices))
    return [image_list[i] for i in ans], [boxes[i] for i in ans]

def generate_filename(dir_name, split_names, images_indices):
    for i, split in enumerate(split_names):
        for im_num in images_indices[i]:
            yield dir_name + '/bar_{}_{}.png'.format(split, str(im_num).zfill(8))

def convert_to_rect_helper(b, index):
    mx1 = min(b[0][index], b[2][index])
    mx2 = min(b[1][index], b[3][index])
    max1 = max(b[0][index], b[2][index])
    max2 = max(b[1][index], b[2][index])
    x1 = min(mx1, mx2)
    x2 = max(max1, max2)
    return slice(x1, x2)

def convert_to_rect(box):
    box = box.astype(np.int32)
    return convert_to_rect_helper(box, 0), convert_to_rect_helper(box, 1)

def rotate(image):
    if image.shape[0] > image.shape[1]:
        rows, cols, channels = image.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
        rot_image = cv2.warpAffine(image, M, (cols, rows))
    # cv2.imshow('try', rot_image)
    # cv2.waitKey(0)
    return rot_image

def write_to_file(filename, word_list, boxes):
    with open(filename, 'w') as fi:
        for word, box in zip(word_list, boxes):
            fi.write('{},{},{},{},{},{},{},{},{}\r\n'.format(word,
                        box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                        ))

def process_images(dir_name, split_names, images_indices, checkpoint_path, crnn_path):
    ### There will be two separate graphs, one for the EAST detection part and another for
    ### the crnn part
    east_graph = tf.Graph()
    with east_graph.as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        f_score, f_geometry = model.model(input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        east_saver = tf.train.Saver(variable_averages.variables_to_restore())

    ## Now the crnn_model
    crnn_graph = tf.Graph()
    with crnn_graph.as_default():
        cropped_image = tf.placeholder(dtype=tf.float32, shape=[1, 32, 100, 3], name='cropped_image')
        word_recog = ShadowNet(phase='Test', hidden_nums=256, layers_nums=2, seq_length=25, num_classes=37)
        with tf.variable_scope('shadow'):
            recog = word_recog.build_shadownet(inputdata=cropped_image)
        decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=recog, sequence_length=25*np.ones(1), merge_repeated=False)
        decoder = data_utils.TextFeatureIO()
        crnn_saver = tf.train.Saver()

    ### loading the checkpoint
    east_session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=east_graph)
    with east_graph.as_default():
        with east_session.as_default():
            ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
            model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            east_saver.restore(east_session, model_path)

    crnn_session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=crnn_graph)
    with crnn_graph.as_default():
        with crnn_session.as_default():
            crnn_saver.restore(crnn_session, save_path=crnn_path)

    for image_name in generate_filename(dir_name, split_names, images_indices):
        print('processing {}'.format(image_name))
        box_list = []
        smaller_image_list = []
        centers = []
        words_list = []
        final_boxes = []
        file_name = image_name.split('.')[0]
        file_name = file_name + '.txt'
        print(image_name)
        im = cv2.imread(image_name)[:, :, ::-1]
        im_resized, (ratio_h, ratio_w) = resize_image(im)
        with east_session.as_default():
            with east_graph.as_default():
                score, geometry = east_session.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
                boxes = detect(score, geometry)
                if boxes is not None:
                    boxes = boxes[:, :8].reshape((-1, 4, 2))
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h
                for box in boxes:
                    box = sort_poly(box.astype(np.int32))
                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                        continue
                    x_range, y_range = convert_to_rect(box)
                    smaller_image = im[y_range, x_range, :]
                    smaller_image_list.append(smaller_image)
                    box_list.append(box)
                    centers.append(((y_range.start + y_range.stop)/2.0, (x_range.start + x_range.stop)/2.0))
        print('East done one the image {}'.format(image_name))


        smaller_images_sorted, box_list = sort_by_pos(smaller_image_list, box_list, centers, im.shape)
        with crnn_session.as_default():
            with crnn_graph.as_default():
                for box, smaller_image in zip(box_list, smaller_images_sorted):
                    smaller_im = cv2.resize(smaller_image, (100, 32))
                    smaller_im = smaller_im[:, :, ::-1]
                    preds = crnn_session.run(decodes, feed_dict={cropped_image:[smaller_im]})
                    preds = decoder.writer.sparse_tensor_to_str(preds[0])
                    if not preds[0] is None:
                        words_list.append(preds[0])
                        final_boxes.append(box)
        print('The words detected are {}'.format(', '.join(words_list)))
        write_to_file(file_name, words_list, final_boxes)
