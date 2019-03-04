

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
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
#from nets.resnet_v1 import resnetv1
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer

#from model.config import cfg
#from model.nms_wrapper import nms
#from model.test import im_detect
#from nets.vgg16 import vgg16
#from nets.resnet_v1 import resnetv1
#from utils.timer import Timer


######################### Newly Added ###########################
#from io import BytesIO
import PIL



CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_60.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

#################### A New Creations ###########################

COLOR = [ 
    [255, 0, 0], [0, 255, 0], [0, 0, 255] 
    ]
color_cnt = 3


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    t1=time.time()*1000
    inds = np.where(dets[:, -1] >= thresh)[0]
    #print("i am go here")
    if len(inds) == 0:
        return
    
    #im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(12, 12))
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ######################### Modified ############################
    ####################### NOT EFFICIENT!!!! #####################
    
    #print("i am running")

    # !!!!!!!!!!!!!!! PYPLOT !!!!!!!!!!!!!!!!! # <------ just ignore it..... one head two big!!!!!!!!!!!!
    
    #ax.imshow(im, aspect='equal')
    #for i in inds:
    #    bbox = dets[i, :4]
    #    score = dets[i, -1]

    #    ax.add_patch(
    #        plt.Rectangle((bbox[0], bbox[1]),
    #                      bbox[2] - bbox[0],
    #                      bbox[3] - bbox[1], fill=False,
    #                      edgecolor='red', linewidth=3.5)
    #    )
    #    ax.text(bbox[0], bbox[1] - 2,
    #            '{:s} {:.3f}'.format(class_name, score),
    #            bbox=dict(facecolor='blue', alpha=0.5),
    #            fontsize=14, color='white')

    #ax.set_title(('{} detections with '
    #              'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                              thresh),
    #             fontsize=14)
    #plt.axis('off')
    #plt.tight_layout()
    #plt.draw()

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ######################## Modified #############################
    
    #plt.savefig("1.jpg")

    ####################### Newly Added ###########################
 
    
    #buffer = BytesIO()
    #plt.savefig(buffer, format = 'png')
    #plt.close()
    #buffer.seek(0)
    #dataPIL = PIL.Image.open(buffer)
    #data = np.asarray(dataPIL)


    ################### AND THEN MODIFIED !! ######################

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1)
        cv2.putText(im, '{:s} {:.3f}'.format(class_name, score), (int(bbox[0]), int(bbox[1])-2), \
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.imshow('Assigned Video',im)
    #buffer.close()
    print('this time, I took' + str(time.time()*1000-t1) + 'ms')
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ######################### Modified ############################
    
    # Load the demo image
    #im_file = os.path.join(cfg.FLAGS2["data_dir"], 'demo', image_name)
    #im = cv2.imread(im_file)
    
    im = image_name    ############### Newly Added ###############

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()

    ############### Modified & Newly Added #######################
    # print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))
    cv2.putText(im, 'Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]), \
                (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    
    # Visualize detections for each class
    CONF_THRESH = 0.1
    NMS_THRESH = 0.1
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        #print("aaa")
        vis_detections(im, cls, dets, thresh=CONF_THRESH)


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
    tfmodel = "F:\\Faster-RCNN\\default\\voc_2007_trainval\\default\\vgg16_faster_rcnn_iter_7600.ckpt"
    #tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
    #                          NETS[demonet][0])

    if not os.path.isfile(tfmodel + '.meta'):
        print(tfmodel)
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture(sess, "TEST", 21,
                            tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

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

    cap = cv2.VideoCapture(0)
    # if you cannot connect to your camera, try to change this value, e.g., change 0 to 1

    while(True):
        if not cap.isOpened():
            print('failed to open stream, reopen this script may correct it')
            break
        ret, img = cap.read()
        cv2.imshow('Proto video', img)

        demo(sess, net, img)
    
        if cv2.waitKey(20)>0:
            break


    #cv2.destroyAllWindow()
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    
    
