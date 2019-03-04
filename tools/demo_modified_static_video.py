

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------




# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Modified by Xueqian Zhang, licensed under the MIT License
# From now on, every change will have a comment with
# 'Newly Added' or 'Modified'
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import _init_paths
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import copy
import json
#from lib.config import config as cfg
#from lib.utils.nms_wrapper import nms
#from lib.utils.test import im_detect
#from nets.resnet_v1 import resnetv1
#from lib.nets.vgg16 import vgg16
#from lib.utils.timer import Timer

from model.config import cfg
from model.nms_wrapper import nms
from model.test import im_detect
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from utils.timer import Timer


######################### Newly Added ###########################
from io import BytesIO
import PIL



#CLASSES = ('__background__',
#           'aeroplane', 'bicycle', 'bird', 'boat',
#           'bottle', 'bus', 'car', 'cat', 'chair',
#           'cow', 'diningtable', 'dog', 'horse',
#           'motorbike', 'person', 'pottedplant',
#           'sheep', 'sofa', 'train', 'tvmonitor')

CLASSES = ('__background__', 'lion', 'monkey', 'panda', 'car', 'person', 'motorbike', 'bicycle')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_60.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

#################### A New Creations ###########################

COLOR = [ 
    [255, 0, 0], [0, 255, 0], [0, 0, 255] 
    ]
color_cnt = 3

frames = {}
cur_ind = 0
glb_ind = 0
tags_name = list()

fourcc = cv2.VideoWriter_fourcc(*'mpeg')

writer = cv2.VideoWriter('/home/sook/code/videoout/temp.avi',fourcc, 8.0, (345, 290))


def vis_detections(im, im2, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    global frames, cur_ind, glb_ind
    t_im = im.copy()
    t_imf = im2.copy()
    t1=time.time()*1000
    tmp_frm = list()
    
    t_n = 1
    for class_name in dets:
        inds = np.where(dets[class_name][:, -1] >= thresh)[0]
    #cv2.imshow('test', im)
        #print('\n', dets[class_name][:, -1], '\n')
        if len(inds) == 0:
            continue
    
    #im = im[:, :, (2, 1, 0)]

        #print('i am here')
        for i in inds:
            bbox = dets[class_name][i, :4]
            score = dets[class_name][i, -1]
            if score < 0.5:
                continue
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1)
            cv2.rectangle(im2, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1)
            cv2.putText(im, '{:s} {:.3f}'.format(class_name, score), (int(bbox[0]), int(bbox[1])-2), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            cv2.putText(im2, '{:s} {:.3f}'.format(class_name, score), (int(bbox[0]), int(bbox[1])-2), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            result = {"x1": int(bbox[0]), "y1": int(bbox[1]), "x2": int(bbox[2]), "y2": int(bbox[3]), 
                      "id": glb_ind, "width": 345, "height": 290, 
                      "type": "Rectangle", "tags": [class_name], "name": t_n}
            if class_name not in tags_name:
                tags_name.append(class_name)
            tmp_frm.append(result)
            t_n += 1
            glb_ind += 1
    if not tmp_frm == []:
        frames[str(cur_ind)] = tmp_frm
        #cv2.imwrite('/home/test/Assigned_video/obj/obj_{:s}.jpg'.format(str(cur_ind).zfill(4)), im)
        #cv2.imwrite('/home/test/Assigned_video/obj_inf/obj_{:s}.jpg'
        #            .format(str(cur_ind).zfill(4)), im2)
        cur_ind += 1
    cv2.imshow('Assigned Video',im)
    writer.write(im)
    cv2.imshow('Assigned inf video', im2)
    #buffer.close()
    print('this time, I took' + str(time.time()*1000-t1) + 'ms')
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
def collect(scores, boxes):
    dets = {}
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        t_dets = np.hstack((cls_boxes,
                            cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(t_dets, 0.1)
        dets[cls] = t_dets[keep, :]
    return dets

def demo(sess, net, image_name, image_name2):
    """Detect object classes in an image using pre-computed object proposals."""
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ######################### Modified ############################
    
    # Load the demo image
    #im_file = os.path.join(cfg.FLAGS2["data_dir"], 'demo', image_name)
    #im = cv2.imread(im_file)
    
    im = image_name    ############### Newly Added ###############
    im_inf = image_name2
    proto_im = image_name.copy()
    proto_imf = image_name2.copy()
 
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()

    ############### Modified & Newly Added #######################
    # print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))
    #cv2.putText(im, 'Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]), \
    #            (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    
    # Visualize detections for each class
    CONF_THRESH = 0.1
    NMS_THRESH = 0.1
    #for cls_ind, cls in enumerate(CLASSES[1:]):
    #    cls_ind += 1  # because we skipped background
    #    cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
    #    cls_scores = scores[:, cls_ind]
    #    dets = np.hstack((cls_boxes,
    #                      cls_scores[:, np.newaxis])).astype(np.float32)
    #    keep = nms(dets, NMS_THRESH)
    #    dets = dets[keep, :]
        #print("aaa")
    dets = collect(scores, boxes)
    
    vis_detections(im, im_inf, dets, thresh=CONF_THRESH)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    # model path
    demonet = 'vgg16'
    #demonet = args.demo_net
    dataset = args.dataset
    #tfmodel = "F:\\Faster-RCNN\\default\\voc_2007_trainval\\default\\vgg16_faster_rcnn_iter_7600.ckpt"
    #tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
    #                          NETS[demonet][0])
    model_path = '/home/sook/code/faster-rcnn-tf/output.bak180810/vgg16/voc_2007_trainval/default/'
    tfmodel = model_path + 'vgg16_faster_rcnn_iter_15000.ckpt'#.data-00000-of-00001'
    #if not os.path.isfile(tfmodel + '.meta'):
    #    print(tfmodel)irtualEnv 
    #    raise IOError(('{:s} not found.\nDid you download the proper networks from '
    #                   'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture("TEST", 8,
                            tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    #saver = tf.train.import_meta_graph(tfmodel + '.meta')
    saver.restore(sess, tfmodel)
    #sess.run(tf.global_variables_initializer())
    
    print('Loaded network {:s}'.format(tfmodel))

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ######################### Modified ############################

    #im_names = ['car001.jpg', 'car002.jpg', 'car003.jpg', 'pika001.jpg',
    #            'pika002.jpg', 'pika003.jpg','kache001.jpg','kache002.jpg','kache003.jpg','kache004.jpg']

    #for im_name in im_names:
    #    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    #    print('Demo for data/demo/{}'.format(im_name))
    #    demo(sess, net, im_name)

    #plt.show()

    ######################### Newly Added ########################

    cap = cv2.VideoCapture('/home/sook/code/infraredAnimals/lion/a7.avi')
    cap_inf = cv2.VideoCapture('/home/sook/code/infraredAnimals/lion/i_a7.avi')
    # if you cannot connect to your camera, try to change this value, e.g., change 0 to 1
    co = True
    ret, img = cap.read()
    while(True):
        if not cap.isOpened():
            print('failed to open stream, reopen this script may correct it')
            break
        if not cap_inf.isOpened():
            print('failed to open stream, reopen this script may correct it')
            break
        if co:
            ret, img = cap.read()
            ret_inf, img_inf = cap_inf.read()
            if not ret: 
                break
            if not ret_inf:
                break
            #cv2.imshow('Proto video', img)
            cv2.imshow('Proto inf video', img_inf)
            co = False
            demo(sess, net, img, img_inf)
        key = cv2.waitKey(20)
        if key == 32:
            break
        if key == 99:
            co = not co
    l = list(map(int, np.linspace(0,cur_ind,num=cur_ind+1,dtype=np.int16)))
    print(l)
    js = {"frames": frames, "framerate": 1, "suggestiontype": "track", "scd": False}
    js["visitedFrames"] = l
    js["inputTags"]=",".join(tags_name)
    #print(js)
    js_str = json.dumps(js)
    #with open('/home/test/Assigned_video/obj_inf.json', 'w') as f:
    #    f.write(js_str)
    print(tags_name)
    cap.release()
    writer.release()
    cap_inf.release()
    cv2.destroyAllWindows()
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    
    
