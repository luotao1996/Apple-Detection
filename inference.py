#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: inference.py
import six
import tensorflow as tf
try:
    import horovod.tensorflow as hvd
except ImportError:
    pass

assert six.PY3, "Apple RCNN requires Python 3!"

from tensorpack import *
from basemodel import (image_preprocess, resnet_c4_backbone, resnet_conv5)
from model_frcnn import (fastrcnn_outputs, fastrcnn_predictions, BoxProposals, FastRCNNHead)
from model_rpn import rpn_head, generate_rpn_proposals
from model_box import (clip_boxes, roi_align, RPNAnchors)
from data import get_all_anchors
from config import config as cfg
from collections import namedtuple
import numpy as np
from common import CustomResize, common_clip_boxes
DetectionResult = namedtuple('DetectionResult',['box', 'score', 'class_id'])

class DetectionModel(ModelDesc):
    def preprocess(self, image):
        image = tf.expand_dims(image, 0)
        image = image_preprocess(image, bgr=True)
        return tf.transpose(image, [0, 3, 1, 2])

    def get_inference_tensor_names(self):
        out = ['output/boxes', 'output/scores', 'output/labels']
        return ['image'], out

    def build_graph(self, *inputs):
        inputs = dict(zip(self.input_names, inputs))

        image = self.preprocess(inputs['image'])     # 1CHW

        features = self.backbone(image)
        anchor_inputs = {k: v for k, v in inputs.items() if k.startswith('anchor_')}
        proposals, rpn_losses = self.rpn(image, features, anchor_inputs)  # inputs?

        targets = [inputs[k] for k in ['gt_boxes', 'gt_labels', 'gt_masks'] if k in inputs]
        self.roi_heads(image, features, proposals, targets)

def detect_one_image(img, model_func):
    """
    Run detection on one image, using the TF callable.
    This function should handle the preprocessing internally.

    Args:
        img: an image
        model_func: a callable from TF model,
            takes image and returns (boxes, probs, labels, [masks])

    Returns:
        [DetectionResult]
    """

    orig_shape = img.shape[:2]
    resizer = CustomResize(cfg.PREPROC.TEST_SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)
    resized_img = resizer.augment(img)
    scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])
    boxes, probs, labels, *_ = model_func(resized_img)
    boxes = boxes / scale
    # boxes are already clipped inside the graph, but after the floating point scaling, this may not be true any more.
    boxes = common_clip_boxes(boxes, orig_shape)
    results = [DetectionResult(*args) for args in zip(boxes, probs, labels)]
    return results



class ResNetC4Model(DetectionModel):
    def inputs(self):
        ret = [
            tf.placeholder(tf.float32, (None, None, 3), 'image'),
            tf.placeholder(tf.int32, (None, None, cfg.RPN.NUM_ANCHOR), 'anchor_labels'),
            tf.placeholder(tf.float32, (None, None, cfg.RPN.NUM_ANCHOR, 4), 'anchor_boxes'),
            tf.placeholder(tf.float32, (None, 4), 'gt_boxes'),
            tf.placeholder(tf.int64, (None,), 'gt_labels')]  # all > 0
        return ret

    def backbone(self, image):
        return [resnet_c4_backbone(image, cfg.BACKBONE.RESNET_NUM_BLOCKS[:3])]

    def rpn(self, image, features, inputs):
        featuremap = features[0]
        rpn_label_logits, rpn_box_logits = rpn_head('rpn', featuremap, cfg.RPN.HEAD_DIM, cfg.RPN.NUM_ANCHOR)
        anchors = RPNAnchors(get_all_anchors(), inputs['anchor_labels'], inputs['anchor_boxes'])
        anchors = anchors.narrow_to(featuremap)

        image_shape2d = tf.shape(image)[2:]     # h,w
        pred_boxes_decoded = anchors.decode_logits(rpn_box_logits)  # fHxfWxNAx4, floatbox
        proposal_boxes, proposal_scores = generate_rpn_proposals(
            tf.reshape(pred_boxes_decoded, [-1, 4]),
            tf.reshape(rpn_label_logits, [-1]),
            image_shape2d,
            cfg.RPN.TEST_PRE_NMS_TOPK,
            cfg.RPN.TEST_POST_NMS_TOPK)
        losses = []
        return BoxProposals(proposal_boxes), losses

    def roi_heads(self, image, features, proposals, targets):
        image_shape2d = tf.shape(image)[2:]     # h,w
        featuremap = features[0]
        gt_boxes, gt_labels, *_ = targets
        boxes_on_featuremap = proposals.boxes * (1.0 / cfg.RPN.ANCHOR_STRIDE)
        roi_resized = roi_align(featuremap, boxes_on_featuremap, 14)
        feature_fastrcnn = resnet_conv5(roi_resized, cfg.BACKBONE.RESNET_NUM_BLOCKS[-1])
        feature_gap = GlobalAvgPooling('gap', feature_fastrcnn, data_format='channels_first')
        fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_outputs('fastrcnn', feature_gap, cfg.DATA.NUM_CLASS)
        fastrcnn_head = FastRCNNHead(proposals, fastrcnn_box_logits, fastrcnn_label_logits, gt_boxes,
                                     tf.constant(cfg.FRCNN.BBOX_REG_WEIGHTS, dtype=tf.float32))
        decoded_boxes = fastrcnn_head.decoded_output_boxes()
        decoded_boxes = clip_boxes(decoded_boxes, image_shape2d, name='fastrcnn_all_boxes')
        label_scores = fastrcnn_head.output_scores(name='fastrcnn_all_scores')
        fastrcnn_predictions(decoded_boxes, label_scores, name_scope='output')

