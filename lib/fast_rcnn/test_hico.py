# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""
import pyximport
pyximport.install()

from fast_rcnn.config import cfg, get_output_dir
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from utils.cython_nms import nms
import cPickle
import heapq
from utils.blob import im_list_to_blob
import os

import scipy.io as sio
from utils.bbox import bbox_overlaps

def _get_extra_bbox(bbox, ind, l=0, u=1):
    overlaps = bbox_overlaps(bbox.astype(np.float), bbox[ind, np.newaxis].astype(np.float))
    over_ind = np.where((overlaps >= l) & (overlaps <= u))[0]
    over_ind = over_ind[ind + 1:]

    rand_ind = np.random.randint(over_ind.shape[0])
    return bbox[ind, np.newaxis], bbox[rand_ind, np.newaxis]

def _get_4_side_bbox(bbox, im_width, im_height):
    assert(bbox.ndim == 1 and bbox.shape[0] == 4)
    # get radius
    w = bbox[2]-bbox[0]+1;
    h = bbox[3]-bbox[1]+1;
    r = (w+h)/2;
    # get boxes
    bbox_l = np.array([np.maximum(bbox[0]-0.5*r,1),
                       bbox[1],
                       bbox[2]-0.5*w,
                       bbox[3]])
    bbox_t = np.array([bbox[0],
                       np.maximum(bbox[1]-0.5*h,1),
                       bbox[2],
                       bbox[3]-0.5*h])
    bbox_r = np.array([bbox[0]+0.5*w,
                       bbox[1],
                       np.minimum(bbox[2]+0.5*r,im_width),
                       bbox[3]])
    bbox_b = np.array([bbox[0],
                       bbox[1]+0.5*h,
                       bbox[2],
                       np.minimum(bbox[3]+0.5*h,im_height)])

    # return in the order left, top, right, bottom
    return bbox_l[None,:], bbox_t[None,:], bbox_r[None,:], bbox_b[None,:]

def _enlarge_bbox(bbox, im_width, im_height):
    w = bbox[2] - bbox[0] + 1;
    h = bbox[3] - bbox[1] + 1;
    r = (w + h) / 2

    x_c = np.floor((bbox[2] + bbox[0]) / 2)
    y_c = np.floor((bbox[3] + bbox[1]) / 2)

    bboxes = np.array([[np.maximum(x_c - w / 2 - r, 1),
                    np.maximum(y_c - h / 2 - r, 1),
                    np.minimum(x_c + w / 2 + r, im_width),
                    np.minimum(y_c + h / 2 + r, im_height)]])

    return bboxes

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    
    # This script is only used for HICO (FLAG_HICO will always be False),
    # so we don't need to handle blob 'rois'
    for ind in xrange(cfg.TOP_K):
        if cfg.FEAT_TYPE == 4:
            rois_l, rois_t, rois_r, rois_b, \
                = _get_4_side_bbox(rois[ind,:], im.shape[1], im.shape[0])
            for s in ['l','t','r','b']:
                key = 'rois_%d_%s' % (ind+1,s)
                # Is this a bug?
                blobs[key] = _get_rois_blob(rois_l, im_scale_factors)
        elif cfg.FLAG_EXTRA:
            im_rois = [0] * 2
            im_rois[0], im_rois[1] \
                = _get_extra_bbox(rois, ind)
            for i in range(2):
                key = 'rois_%d' % (ind * 2 + i + 1)
                blobs[key] = _get_rois_blob(im_rois[i], im_scale_factors)
        else:
            key = 'rois_%d' % (ind+1)
            if cfg.FLAG_ENLARGE:
                rois_e = _enlarge_bbox(rois[ind, :], im.shape[1], im.shape[0])
                blobs[key] = _get_rois_blob(rois_e, im_scale_factors)
            else:
                blobs[key] = _get_rois_blob(rois[ind:ind+1,:], im_scale_factors)

    return blobs, im_scale_factors

# def _bbox_pred(boxes, box_deltas):
#     """Transform the set of class-agnostic boxes into class-specific boxes
#     by applying the predicted offsets (box_deltas)
#     """
#     if boxes.shape[0] == 0:
#         return np.zeros((0, box_deltas.shape[1]))
# 
#     boxes = boxes.astype(np.float, copy=False)
#     widths = boxes[:, 2] - boxes[:, 0] + cfg.EPS
#     heights = boxes[:, 3] - boxes[:, 1] + cfg.EPS
#     ctr_x = boxes[:, 0] + 0.5 * widths
#     ctr_y = boxes[:, 1] + 0.5 * heights
# 
#     dx = box_deltas[:, 0::4]
#     dy = box_deltas[:, 1::4]
#     dw = box_deltas[:, 2::4]
#     dh = box_deltas[:, 3::4]
# 
#     pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
#     pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
#     pred_w = np.exp(dw) * widths[:, np.newaxis]
#     pred_h = np.exp(dh) * heights[:, np.newaxis]
# 
#     pred_boxes = np.zeros(box_deltas.shape)
#     # x1
#     pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
#     # y1
#     pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
#     # x2
#     pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
#     # y2
#     pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
# 
#     return pred_boxes

# def _clip_boxes(boxes, im_shape):
#     """Clip boxes to image boundaries."""
#     # x1 >= 0
#     boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
#     # y1 >= 0
#     boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
#     # x2 < im_shape[1]
#     boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
#     # y2 < im_shape[0]
#     boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
#     return boxes

def im_detect(net, im, boxes):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    print boxes.shape;
    blobs, unused_im_scale_factors = _get_blobs(im, boxes)

    # Disable box dedup for HICO
    # # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # # (some distinct image ROIs get mapped to the same feature ROI).
    # # Here, we identify duplicate feature ROIs, so we only compute features
    # # on the unique subset.
    # if cfg.DEDUP_BOXES > 0:
    #     v = np.array([1, 1e3, 1e6, 1e9, 1e12])
    #     hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
    #     _, index, inv_index = np.unique(hashes, return_index=True,
    #                                     return_inverse=True)
    #     blobs['rois'] = blobs['rois'][index, :]
    #     boxes = boxes[index, :]

    # reshape network inputs and concat input blobs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    for ind in xrange(cfg.TOP_K):
        if cfg.FEAT_TYPE == 4:
            for s in ['l','t','r','b']:
                key = 'rois_%d_%s' % (ind+1,s)
                net.blobs[key].reshape(*(blobs[key].shape))
        elif cfg.FLAG_EXTRA:
            for i in range(2):
                key = 'rois_%d' % (ind * 2 + i + 1)
                net.blobs[key].reshape(*(blobs[key].shape))
        else:
            key = 'rois_%d' % (ind+1)
            net.blobs[key].reshape(*(blobs[key].shape))

    # forward pass    
    blobs_out = net.forward(**(blobs))

    # if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        # scores = net.blobs['cls_score'].data
    # else:
        # use softmax estimated probabilities
        # scores = blobs_out['cls_prob']
    scores = []
    
    # save feature
    if cfg.FEAT_TYPE == 4 and not cfg.FLAG_SIGMOID:
        feats = net.blobs['fc7_concat'].data
    elif cfg.FLAG_EXTRA:
        feats = net.blobs['cls_score_sum'].data
    elif cfg.FLAG_SIGMOID:
        feats = net.blobs['cls_score'].data
    else:
        feats = net.blobs['fc7'].data

    # assert(cfg.TEST.BBOX_REG == False)
    # if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        # box_deltas = blobs_out['bbox_pred']
        # pred_boxes = _bbox_pred(boxes, box_deltas)
        # pred_boxes = _clip_boxes(pred_boxes, im.shape)
    # else:
        # Simply repeat the boxes, once for each class
        # pred_boxes = np.tile(boxes, (1, scores.shape[1]))
    pred_boxes = []

    # Disable box dedup for HICO
    # if cfg.DEDUP_BOXES > 0:
    #     # Map scores and predictions back to the original set of boxes
    #     scores = scores[inv_index, :]
    #     pred_boxes = pred_boxes[inv_index, :]
    #     feats = feats[inv_index, :]

    return scores, pred_boxes, feats

# def vis_detections(im, class_name, dets, thresh=0.3):
#     """Visual debugging of detections."""
#     import matplotlib.pyplot as plt
#     im = im[:, :, (2, 1, 0)]
#     for i in xrange(np.minimum(10, dets.shape[0])):
#         bbox = dets[i, :4]
#         score = dets[i, -1]
#         if score > thresh:
#             plt.cla()
#             plt.imshow(im)
#             plt.gca().add_patch(
#                 plt.Rectangle((bbox[0], bbox[1]),
#                               bbox[2] - bbox[0],
#                               bbox[3] - bbox[1], fill=False,
#                               edgecolor='g', linewidth=3)
#                 )
#             plt.title('{}  {:.3f}'.format(class_name, score))
#             plt.show()

# def apply_nms(all_boxes, thresh):
#     """Apply non-maximum suppression to all predicted boxes output by the
#     test_net method.
#     """
#     num_classes = len(all_boxes)
#     num_images = len(all_boxes[0])
#     nms_boxes = [[[] for _ in xrange(num_images)]
#                  for _ in xrange(num_classes)]
#     for cls_ind in xrange(num_classes):
#         for im_ind in xrange(num_images):
#             dets = all_boxes[cls_ind][im_ind]
#             if dets == []:
#                 continue
#             keep = nms(dets, thresh)
#             if len(keep) == 0:
#                 continue
#             nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
#     return nms_boxes

def test_net_hico(net, imdb, feat_root):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # # heuristic: keep an average of 40 detections per class per images prior
    # # to NMS
    # max_per_set = 40 * num_images
    # # heuristic: keep at most 100 detection per class per image prior to NMS
    # max_per_image = 100
    # # detection thresold for each class (this is adaptively set based on the
    # # max_per_set constraint)
    # thresh = -np.inf * np.ones(imdb.num_classes)
    # # top_scores will hold one minheap of scores per class (used to enforce
    # # the max_per_set constraint)
    # top_scores = [[] for _ in xrange(imdb.num_classes)]
    # # all detections are collected into:
    # #    all_boxes[cls][image] = N x 5 array of detections in
    # #    (x1, y1, x2, y2, score)
    # all_boxes = [[[] for _ in xrange(num_images)]
    #              for _ in xrange(imdb.num_classes)]

    # output_dir = get_output_dir(imdb, net)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    
    if not os.path.exists(feat_root):
        os.makedirs(feat_root)

    assert(cfg.FLAG_HICO == True)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    roidb = imdb.roidb
    for i in xrange(num_images):
        im_name = os.path.splitext(os.path.basename(imdb.image_path_at(i)))[0]
        feat_file = os.path.join(feat_root, im_name + '.mat')
        if os.path.isfile(feat_file):
            continue

        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()
        scores, boxes, feats = im_detect(net, im, roidb[i]['boxes'])
        _t['im_detect'].toc()

        _t['misc'].tic()
        # for j in xrange(1, imdb.num_classes):
        #     inds = np.where((scores[:, j] > thresh[j]) &
        #                     (roidb[i]['gt_classes'] == 0))[0]
        #     cls_scores = scores[inds, j]
        #     cls_boxes = boxes[inds, j*4:(j+1)*4]
        #     top_inds = np.argsort(-cls_scores)[:max_per_image]
        #     cls_scores = cls_scores[top_inds]
        #     cls_boxes = cls_boxes[top_inds, :]
        #     # push new scores onto the minheap
        #     for val in cls_scores:
        #         heapq.heappush(top_scores[j], val)
        #     # if we've collected more than the max number of detection,
        #     # then pop items off the minheap and update the class threshold
        #     if len(top_scores[j]) > max_per_set:
        #         while len(top_scores[j]) > max_per_set:
        #             heapq.heappop(top_scores[j])
        #         thresh[j] = top_scores[j][0]

        #     all_boxes[j][i] = \
        #             np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
        #             .astype(np.float32, copy=False)

        #     if 0:
        #         keep = nms(all_boxes[j][i], 0.3)
        #         vis_detections(im, imdb.classes[j], all_boxes[j][i][keep, :])
        sio.savemat(feat_file, {'feat' : feats})
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    # for j in xrange(1, imdb.num_classes):
    #     for i in xrange(num_images):
    #         inds = np.where(all_boxes[j][i][:, -1] > thresh[j])[0]
    #         all_boxes[j][i] = all_boxes[j][i][inds, :]

    # # det_file = os.path.join(output_dir, 'detections.pkl')
    # with open(det_file, 'wb') as f:
    #     cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    # print 'Applying NMS to all detections'
    # nms_dets = apply_nms(all_boxes, cfg.TEST.NMS)

    # print 'Evaluating detections'
    # imdb.evaluate_detections(nms_dets, output_dir)
