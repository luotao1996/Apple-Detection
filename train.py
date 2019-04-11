#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py

import os
import argparse
import six
import tensorflow as tf
try:
    import horovod.tensorflow as hvd
except ImportError:
    pass

assert six.PY3, "Apple RCNN requires Python 3!"

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils import optimizer
from basemodel import (image_preprocess, resnet_c4_backbone, resnet_conv5)
from model_frcnn import (sample_fast_rcnn_targets, fastrcnn_outputs, BoxProposals, FastRCNNHead)
from model_rpn import rpn_head, rpn_losses, generate_rpn_proposals
from model_box import (roi_align, RPNAnchors)
from data import (get_train_dataflow, get_all_anchors)
from config import finalize_configs, config as cfg


class DetectionModel(ModelDesc):
    def preprocess(self, image):
        image = tf.expand_dims(image, 0)
        image = image_preprocess(image, bgr=True)
        return tf.transpose(image, [0, 3, 1, 2])

    @property
    def training(self):
        return get_current_tower_context().is_training

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.003, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)

        # The learning rate is set for 8 GPUs, and we use trainers with average=False.
        lr = lr / 8.
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        if cfg.TRAIN.NUM_GPUS < 8:
            opt = optimizer.AccumGradOptimizer(opt, 8 // cfg.TRAIN.NUM_GPUS)
        return opt

    def get_inference_tensor_names(self):
        """
        Returns two lists of tensor names to be used to create an inference callable.

        Returns:
            [str]: input names
            [str]: output names
        """
        out = ['output/boxes', 'output/scores', 'output/labels']
        return ['image'], out

    def build_graph(self, *inputs):
        inputs = dict(zip(self.input_names, inputs))

        image = self.preprocess(inputs['image'])     # 1CHW

        features = self.backbone(image)
        anchor_inputs = {k: v for k, v in inputs.items() if k.startswith('anchor_')}
        proposals, rpn_losses = self.rpn(image, features, anchor_inputs)  # inputs?

        targets = [inputs[k] for k in ['gt_boxes', 'gt_labels', 'gt_masks'] if k in inputs]
        head_losses = self.roi_heads(image, features, proposals, targets)

        wd_cost = regularize_cost(
            '.*/W', l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), name='wd_cost')
        total_cost = tf.add_n(
            rpn_losses + head_losses + [wd_cost], 'total_cost')
        add_moving_summary(total_cost, wd_cost)
        return total_cost

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
            cfg.RPN.TRAIN_PRE_NMS_TOPK,
            cfg.RPN.TRAIN_POST_NMS_TOPK)
        losses = rpn_losses(
            anchors.gt_labels, anchors.encoded_gt_boxes(), rpn_label_logits, rpn_box_logits)
        return BoxProposals(proposal_boxes), losses

    def roi_heads(self, image, features, proposals, targets):
        featuremap = features[0]
        gt_boxes, gt_labels, *_ = targets
        proposals = sample_fast_rcnn_targets(proposals.boxes, gt_boxes, gt_labels)
        boxes_on_featuremap = proposals.boxes * (1.0 / cfg.RPN.ANCHOR_STRIDE)
        roi_resized = roi_align(featuremap, boxes_on_featuremap, 14)

        feature_fastrcnn = resnet_conv5(roi_resized, cfg.BACKBONE.RESNET_NUM_BLOCKS[-1])    # nxcx7x7
        # Keep C5 feature to be shared with mask branch
        feature_gap = GlobalAvgPooling('gap', feature_fastrcnn, data_format='channels_first')
        fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_outputs('fastrcnn', feature_gap, cfg.DATA.NUM_CLASS)
        fastrcnn_head = FastRCNNHead(proposals, fastrcnn_box_logits, fastrcnn_label_logits, gt_boxes,
                                     tf.constant(cfg.FRCNN.BBOX_REG_WEIGHTS, dtype=tf.float32))
        return fastrcnn_head.losses()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load a model for evaluation or training. Can overwrite BACKBONE.WEIGHTS')
    parser.add_argument('--logdir', help='log directory', default='train_log/maskrcnn')
    parser.add_argument('--config', help="A list of KEY=VALUE to overwrite those defined in config.py",
                        nargs='+')
    args = parser.parse_args()
    if args.config:
        cfg.update_args(args.config)

    MODEL = ResNetC4Model()
    is_horovod = cfg.TRAINER == 'horovod'
    if is_horovod:
        hvd.init()
        logger.info("Horovod Rank={}, Size={}".format(hvd.rank(), hvd.size()))

    if not is_horovod or hvd.rank() == 0:
        logger.set_logger_dir(args.logdir, 'd')

    finalize_configs(is_training=True)
    stepnum = cfg.TRAIN.STEPS_PER_EPOCH

    # warmup is step based, lr is epoch based
    init_lr = cfg.TRAIN.BASE_LR * 0.33 * min(8. / cfg.TRAIN.NUM_GPUS, 1.)
    warmup_schedule = [(0, init_lr), (cfg.TRAIN.WARMUP, cfg.TRAIN.BASE_LR)]
    warmup_end_epoch = cfg.TRAIN.WARMUP * 1. / stepnum
    lr_schedule = [(int(warmup_end_epoch + 0.5), cfg.TRAIN.BASE_LR)]

    factor = 8. / cfg.TRAIN.NUM_GPUS
    for idx, steps in enumerate(cfg.TRAIN.LR_SCHEDULE[:-1]):
        mult = 0.1 ** (idx + 1)
        lr_schedule.append(
            (steps * factor // stepnum, cfg.TRAIN.BASE_LR * mult))
    logger.info("Warm Up Schedule (steps, value): " + str(warmup_schedule))
    logger.info("LR Schedule (epochs, value): " + str(lr_schedule))
    train_dataflow = get_train_dataflow()
    # This is what's commonly referred to as "epochs"
    total_passes = cfg.TRAIN.LR_SCHEDULE[-1] * 8 / train_dataflow.size()
    logger.info("Total passes of the training set is: {:.5g}".format(total_passes))

    callbacks = [
        PeriodicCallback(
            ModelSaver(max_to_keep=10, keep_checkpoint_every_n_hours=1),
            every_k_epochs=20),
        # linear warmup
        ScheduledHyperParamSetter(
            'learning_rate', warmup_schedule, interp='linear', step_based=True),
        ScheduledHyperParamSetter('learning_rate', lr_schedule),
        PeakMemoryTracker(),
        EstimatedTimeLeft(median=True),
        SessionRunTimeout(60000).set_chief_only(True),   # 1 minute timeout
    ]
    if not is_horovod:
        callbacks.append(GPUUtilizationTracker())

    if is_horovod and hvd.rank() > 0:
        session_init = None
    else:
        if args.load:
            session_init = get_model_loader(args.load)
        else:
            session_init = get_model_loader(cfg.BACKBONE.WEIGHTS) if cfg.BACKBONE.WEIGHTS else None

    traincfg = TrainConfig(
        model=MODEL,
        data=FeedInput(train_dataflow),
        callbacks=callbacks,
        steps_per_epoch=stepnum,
        max_epoch=cfg.TRAIN.LR_SCHEDULE[-1] * factor // stepnum,
        session_init=session_init,
    )
    if is_horovod:
        trainer = HorovodTrainer(average=False)
    else:
        # nccl mode appears faster than cpu mode
        trainer = SyncMultiGPUTrainerReplicated(cfg.TRAIN.NUM_GPUS, average=False, mode='nccl')
    launch_train_with_config(traincfg, trainer)
