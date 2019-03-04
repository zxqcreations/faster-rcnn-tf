from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

import numpy as np

from layer_utils.snippets import generate_anchors_pre, generate_anchors_pre_tf
from layer_utils.proposal_layer import proposal_layer, proposal_layer_tf
from layer_utils.proposal_top_layer import proposal_top_layer, proposal_top_layer_tf
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer
from utils.visualization import draw_bounding_boxes

from model.config import cfg
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import gfile


class redev():
    def __init__(self, ckpt_file):
        self.ckpt = ckpt_file
        self._feat_stride = [16, ]
        self._feat_compress = [1. / float(self._feat_stride[0]), ]
        self._scope = 'vgg_16'
        self._layers = {}
        self._predictions = {}
        self.con = None
    
    def _get_variables_in_checkpoint_file(self):
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(self.ckpt)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return var_to_shape_map 
        except Exception as e:  # pylint: disable=broad-except
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print("It's likely that your checkpoint file has been compressed "
                    "with SNAPPY.")
    
    def _image_to_head(self, reuse=None):
        with tf.variable_scope(self._scope, self._scope, reuse=reuse):
            net = slim.repeat(self._image, 1, slim.conv2d, 32, [3, 3],
                                trainable=False, scope='conv1')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
            net = slim.repeat(net, 1, slim.conv2d, 64, [3, 3],
                                trainable=False, scope='conv2')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
            net = slim.repeat(net, 1, slim.conv2d, 64, [3, 3],
                                trainable=False, scope='conv3')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
            net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3],
                                trainable=False, scope='conv4')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
            net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3],
                                trainable=False, scope='conv5')

        #self._act_summaries.append(net)
        self._layers['head'] = net
        
        return net
    
    def _head_to_tail(self, pool5, reuse=None):
        with tf.variable_scope(self._scope, self._scope, reuse=reuse):
            pool5_flat = slim.flatten(pool5, scope='flatten')
            fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
            fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
        return fc7
    
    def _anchor_component(self):
        with tf.variable_scope('ANCHOR_' + self._tag) as scope:
            # just to get the shape right
            height = tf.to_int32(tf.ceil(self._im_info[0] / np.float32(self._feat_stride[0])))
            width = tf.to_int32(tf.ceil(self._im_info[1] / np.float32(self._feat_stride[0])))
            if cfg.USE_E2E_TF:
                anchors, anchor_length = generate_anchors_pre_tf(
                                        height,
                                        width,
                                        self._feat_stride,
                                        self._anchor_scales,
                                        self._anchor_ratios
                                         )
            else:
                anchors, anchor_length = tf.py_func(generate_anchors_pre,
                                                [height, width,
                                                 self._feat_stride, self._anchor_scales, 
                                                 self._anchor_ratios],
                                                [tf.float32, tf.int32], name="generate_anchors")
            anchors.set_shape([None, 4])
            anchor_length.set_shape([])
            self._anchors = anchors
            self._anchor_length = anchor_length
    
    
    
    def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name) as scope:
            if cfg.USE_E2E_TF:
                rois, rpn_scores = proposal_layer_tf(
                        rpn_cls_prob,
                        rpn_bbox_pred,
                        self._im_info,
                        self._mode,
                        self._feat_stride,
                        self._anchors,
                        self._num_anchors
                        )
            else:
                rois, rpn_scores = tf.py_func(proposal_layer,
                                  [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                                   self._feat_stride, self._anchors, self._num_anchors],
                                  [tf.float32, tf.float32], name="proposal")

            rois.set_shape([None, 5])
            rpn_scores.set_shape([None, 1])

        return rois, rpn_scores
        
    def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name) as scope:
            if cfg.USE_E2E_TF:
                rois, rpn_scores = proposal_top_layer_tf(
                    rpn_cls_prob,
                    rpn_bbox_pred,
                    self._im_info,
                    self._feat_stride,
                    self._anchors,
                    self._num_anchors
                    )
            else:
                rois, rpn_scores = tf.py_func(proposal_top_layer,
                                  [rpn_cls_prob, rpn_bbox_pred, self._im_info,
                                   self._feat_stride, self._anchors, self._num_anchors],
                                  [tf.float32, tf.float32], name="proposal_top")
            
            rois.set_shape([cfg.TEST.RPN_TOP_N, 5])
            rpn_scores.set_shape([cfg.TEST.RPN_TOP_N, 1])

            return rois, rpn_scores
    
    def _reshape_layer(self, bottom, num_dim, name):
        input_shape = tf.shape(bottom)
        with tf.variable_scope(name) as scope:
            # change the channel to the caffe format
            to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
            # then force it to have channel 2
            reshaped = tf.reshape(to_caffe,
                                tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
            # then swap the channel back
            to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
        return to_tf
            
    def _softmax_layer(self, bottom, name):
        if name.startswith('rpn_cls_prob_reshape'):
            input_shape = tf.shape(bottom)
            bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
            reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
            return tf.reshape(reshaped_score, input_shape)
        return tf.nn.softmax(bottom, name=name)
        
    def _region_proposal(self, net_conv, initializer):
        rpn = slim.conv2d(net_conv, cfg.RPN_CHANNELS, [3, 3], trainable=False, 
                        weights_initializer=initializer,
                        scope="rpn_conv/3x3")
        #self._act_summaries.append(rpn)
        
        # change it so that the score has 2 as its channel size
        rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=False,
                                weights_initializer=initializer,
                                padding='VALID', activation_fn=None, scope='rpn_cls_score')
        # change it so that the score has 2 as its channel size
        rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
        rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
        
        rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
        rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=False,
                                    weights_initializer=initializer,
                                    padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
        if cfg.TEST.MODE == 'nms':
            rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        elif cfg.TEST.MODE == 'top':
            rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        else:
            raise NotImplementedError

        
        self._predictions["rois"] = rois

        return rois
    
    def _crop_pool_layer(self, bottom, rois, name):
        with tf.variable_scope(name) as scope:
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            # Get the normalized coordinates of bounding boxes
            bottom_shape = tf.shape(bottom)
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
            # Won't be back-propagated to rois anyway, but to save time
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            pre_pool_size = cfg.POOLING_SIZE * 2
            crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")

        return slim.max_pool2d(crops, [2, 2], padding='SAME')
    
    def _region_classification(self, fc7, initializer, initializer_bbox):
        cls_score = slim.fully_connected(fc7, self._num_classes, 
                                           weights_initializer=initializer,
                                           trainable=False,
                                           activation_fn=None, scope='cls_score')
        cls_prob = self._softmax_layer(cls_score, "cls_prob")
        cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")
        bbox_pred = slim.fully_connected(fc7, self._num_classes * 4, 
                                         weights_initializer=initializer_bbox,
                                         trainable=False,
                                         activation_fn=None, scope='bbox_pred')

        self._predictions["cls_score"] = cls_score
        self._predictions["cls_pred"] = cls_pred
        self._predictions["cls_prob"] = cls_prob
        self._predictions["bbox_pred"] = bbox_pred

        return cls_prob, bbox_pred
    
    def _build_network(self, is_training=True):
        # select initializers
        if cfg.TRAIN.TRUNCATED:
            initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
        else:
            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

        net_conv = self._image_to_head()
        with tf.variable_scope(self._scope, self._scope):
            # build the anchors for the image
            self._anchor_component()
            # region proposal network
            rois = self._region_proposal(net_conv, initializer)
            # region of interest pooling
            if cfg.POOLING_MODE == 'crop':
                pool5 = self._crop_pool_layer(net_conv, rois, "pool5")
            else:
                raise NotImplementedError

        fc7 = self._head_to_tail(pool5)
        with tf.variable_scope(self._scope, self._scope):
            # region classification
            cls_prob, bbox_pred = self._region_classification(fc7, initializer, initializer_bbox)

        #self._score_summaries.update(self._predictions)

        return rois, cls_prob, bbox_pred
    
    
    def create_architecture(self, mode, num_classes, tag=None, 
                            anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
        self._image = tf.placeholder(tf.float32, shape=[1, 600, 712, 3], name='input')
        self._im_info = tf.constant([600, 712, 2.0833333], dtype=tf.float32)
        self._tag = tag

        self._num_classes = num_classes
        self._mode = mode
        self._anchor_scales = anchor_scales
        self._num_scales = len(anchor_scales)

        self._anchor_ratios = anchor_ratios
        self._num_ratios = len(anchor_ratios)

        self._num_anchors = self._num_scales * self._num_ratios
        # handle most of the regularizers here
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
        if cfg.TRAIN.BIAS_DECAY:
            biases_regularizer = weights_regularizer
        else:
            biases_regularizer = tf.no_regularizer

        # list as many types of layers as possible, even if they are not used now
        with arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                        slim.conv2d_transpose, slim.separable_conv2d, \
                        slim.fully_connected], 
                        weights_regularizer=weights_regularizer,
                        biases_regularizer=biases_regularizer, 
                        biases_initializer=tf.constant_initializer(0.0)): 
            rois, cls_prob, bbox_pred = self._build_network()
        with tf.variable_scope('concat') as scope:
            self.con = tf.concat([self._predictions["cls_score"], 
                                  self._predictions['cls_prob'], 
                                  self._predictions['bbox_pred'], 
                                  self._predictions['rois']], 1, name='concat')
        a = tf.global_variables()
        for i in a:
            print(i)
        
    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []
        
        for v in variables:
            if v.name.split(':')[0] in var_keep_dic:
                variables_to_restore.append(v)
        return variables_to_restore
        
    def construct_graph(self, sess):
        with sess.graph.as_default():
            self.create_architecture('TEST', 8,
                            tag='default', anchor_scales=[8, 16, 32]) 
    
    def restore_and_redev(self):
        #self.construct_graph(sess)
        var_ckpt = self._get_variables_in_checkpoint_file()
        var_mdl = tf.global_variables()
        var_to_restore = self.get_variables_to_restore(var_mdl, var_ckpt)
        variables = tf.global_variables()
        with tf.Session() as sess:
            sess.run(tf.variables_initializer(variables, name='init'))
            #restorer = tf.train.Saver(var_to_restore)
            #restorer.restore(sess, self.ckpt)
            print('restored!')
            #with tf.variable_scope('concat') as scope:
            #    self.con = tf.concat([self._predictions["cls_score"], 
            #                          self._predictions['cls_prob'], 
             #                         self._predictions['bbox_pred'], 
            #                          self._predictions['rois']], 1, name='concat')
            with open('graph.txt', 'w') as f:
                f.write(str(tf.get_default_graph().as_graph_def()))
            with gfile.GFile('/home/sook/code/faster-rcnn-tf/vgg16_faster_rcnn.pb', 'wb') as f:
                f.write(tf.get_default_graph().as_graph_def().SerializeToString())
        #with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.save(sess,'/home/sook/code/faster-rcnn-tf/new_ckpt/vgg16_frcnn_new.ckpt')
            print('saved!')


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
