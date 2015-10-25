# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import pyximport
pyximport.install()

import random
import numpy as np
import numpy.random as npr
import cv2
from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
from utils.cython_bbox import bbox_overlaps

def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    # Now, build the region of interest and label blobs
    rois_blob = np.zeros((0, 5), dtype=np.float32)
    labels_blob = np.zeros((0), dtype=np.float32)
    bbox_targets_blob = np.zeros((0, 4 * num_classes), dtype=np.float32)
    bbox_loss_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
    # all_overlaps = []
    for im_i in xrange(num_images):
        labels, overlaps, im_rois, bbox_targets, bbox_loss \
            = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image,
                           num_classes)

        # Add to RoIs blob
        rois = _project_im_rois(im_rois, im_scales[im_i])
        batch_ind = im_i * np.ones((rois.shape[0], 1))
        rois_blob_this_image = np.hstack((batch_ind, rois))
        rois_blob = np.vstack((rois_blob, rois_blob_this_image))

        # Add to labels, bbox targets, and bbox loss blobs
        labels_blob = np.hstack((labels_blob, labels))
        bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
        bbox_loss_blob = np.vstack((bbox_loss_blob, bbox_loss))
        # all_overlaps = np.hstack((all_overlaps, overlaps))

    # For debug visualizations
    # _vis_minibatch(im_blob, rois_blob, labels_blob, all_overlaps)

    blobs = {'data': im_blob,
             'rois': rois_blob,
             'labels': labels_blob}

    if cfg.TRAIN.BBOX_REG:
        blobs['bbox_targets'] = bbox_targets_blob
        blobs['bbox_loss_weights'] = bbox_loss_blob

    return blobs

def get_minibatch_bbox(roidb):
    num_images = len(roidb)
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
        size=num_images)

    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)

    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    rois_blob = np.zeros((0, 5), dtype=np.float32)
    labels_blob = np.zeros((0, 4), dtype=np.float32)

    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)
    rois_blob = np.zeros((0, 5), dtype=np.float32)
    labels_blob = np.zeros((0, 4), dtype=np.float32)

    for im_ind in range(num_images):
        labels, im_rois = _sample_rois_bbox(roidb[im_ind], rois_per_image)

        rois = _project_im_rois(im_rois, im_scales[im_ind])
        batch_ind = im_ind * np.ones((rois.shape[0], 1))
        rois_blob_this_image = np.hstack((batch_ind, rois))
        rois_blob = np.vstack((rois_blob, rois_blob_this_image))

        labels_blob = np.vstack((labels_blob, labels))
            # cv_im = im.copy()
            # cv2.rectangle(cv_im, (int(rois[0, 0]), int(rois[0, 1])), (int(rois[0, 2]), int(rois[0, 3])), (0, 255, 0))

            # im_w = rois[0, 2] - rois[0, 0]
            # im_h = rois[0, 3] - rois[0, 1]

            # x_c = im_w * labels[j][0, 0]
            # y_c = im_h * labels[j][0, 1]

            # x1 = int(x_c - im_w * labels[j][0, 2] / 2)
            # y1 = int(y_c - im_h * labels[j][0, 3] / 2)
            # x2 = int(x_c + im_w * labels[j][0, 2] / 2)
            # y2 = int(y_c + im_h * labels[j][0, 3] / 2)
            # cv2.rectangle(cv_im, (x1, y1), (x2, y2), (0, 0, 255))

            # cv2.imshow('test', cv_im)
            # print labels[j]
            # raw_input('Press Enter to Continue')

    blobs = {'data': im_blob,
             'rois': rois_blob,
             'labels': labels_blob}

    return blobs

def _sample_rois_bbox(roi, rois_per_image, ov_threshold=0.6):
    gt_inds = np.where(roi['gt_classes'])[0]
    overlaps = bbox_overlaps(roi['boxes'].astype(np.float), 
                    roi['boxes'][gt_inds, :].astype(np.float))

    roi_im = cv2.imread(roi['image'])
    im_rois = np.zeros((0, 4), dtype=np.float32)
    labels = np.zeros((0, 4), dtype=np.float32)
    for k in range(overlaps.shape[1]):
        gt_box = roi['boxes'][gt_inds[k], :]
        overlap = overlaps[:, k]
        ov_inds = np.where(overlap >= ov_threshold)[0]

        im_roi = _enlarge_boxes(roi['boxes'][ov_inds, :], 
                    roi_im.shape[1], roi_im.shape[0])
        label = _sample_label(im_roi, gt_box)

        im_rois = np.vstack((im_rois, im_roi))
        labels = np.vstack((labels, label))

    rois_inds = npr.permutation(im_rois.shape[0])
    if rois_per_image < im_rois.shape[0]:
        rois_inds = rois_inds[0:rois_per_image]

    print labels.shape
    print rois_inds
    labels = labels[rois_inds]
    im_rois = im_rois[rois_inds]

    return labels, im_rois

def _sample_label(boxes, gt_box):
    w = (boxes[:, 2] - boxes[:, 0] + 1)[:, np.newaxis]
    h = (boxes[:, 3] - boxes[:, 1] + 1)[:, np.newaxis]

    w_gt = gt_box[2] - gt_box[0] + 1
    h_gt = gt_box[3] - gt_box[1] + 1
    x_gt = np.floor((gt_box[0] + w_gt / 2 - boxes[:, 0])[:, np.newaxis])
    y_gt = np.floor((gt_box[1] + h_gt / 2 - boxes[:, 1])[:, np.newaxis])

    sub_label = np.hstack((x_gt / w, y_gt / h, w_gt / w, h_gt / h))
    return sub_label

def _enlarge_boxes(boxes, im_width, im_height):
    w = (boxes[:, 2] - boxes[:, 0] + 1)[:, np.newaxis]
    h = (boxes[:, 3] - boxes[:, 1] + 1)[:, np.newaxis]
    # r = (w + h) / 4
    w = 1.25 * w
    h = 1.25 * h

    x_c = np.floor((boxes[:, 2] + boxes[:, 0]) / 2)[:, np.newaxis]
    y_c = np.floor((boxes[:, 3] + boxes[:, 1]) / 2)[:, np.newaxis]

    # e_boxes = np.hstack((np.maximum(x_c - w / 2 - r, 1),
        # np.maximum(y_c - h / 2 - r, 1),
        # np.minimum(x_c + w / 2 + r, im_width),
        # np.minimum(y_c + h / 2 + r, im_height)))
    e_boxes = np.hstack((np.maximum(x_c - w / 2, 0),
        np.maximum(y_c - h / 2, 0),
        np.minimum(x_c + w / 2, im_width),
        np.minimum(y_c + h / 2, im_height)))
    
    return e_boxes

def _sample_rois(roidb, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # label = class RoI has max overlap with
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']
    rois = roidb['boxes']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image,
                             replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image,
                             replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    overlaps = overlaps[keep_inds]
    rois = rois[keep_inds]

    bbox_targets, bbox_loss_weights = \
            _get_bbox_regression_labels(roidb['bbox_targets'][keep_inds, :],
                                        num_classes)

    return labels, overlaps, rois, bbox_targets, bbox_loss_weights

def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales

def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_loss_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_loss_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_loss_weights[ind, start:end] = [1., 1., 1., 1.]
    return bbox_targets, bbox_loss_weights

def _vis_minibatch(im_blob, rois_blob, labels_blob, overlaps):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    for i in xrange(rois_blob.shape[0]):
        rois = rois_blob[i, :]
        im_ind = rois[0]
        roi = rois[1:]
        im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        cls = labels_blob[i]
        plt.imshow(im)
        print 'class: ', cls, ' overlap: ', overlaps[i]
        plt.gca().add_patch(
            plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor='r', linewidth=3)
            )
        plt.show()
