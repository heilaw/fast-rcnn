from fast_rcnn.config import cfg, get_output_dir
from utils.blob import im_list_to_blob
from utils.timer import Timer
from utils.rois import enlarge_rois

import cv2
import os.path
import numpy as np
import cPickle

def _get_image_blob(im):
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    blob = im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _get_blobs(im, rois):
    blobs = {'data': None, 'rois': None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    
    im_rois = enlarge_rois(rois, im.shape[1], im.shape[0])
    blobs['rois'] = _get_rois_blob(im_rois, im_scale_factors)

    return blobs, im_scale_factors

def _project_im_rois(im_rois, scales):
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] = im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argsort(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]
    return rois, levels

def bbox_detect(net, im, boxes):
    """
    Generate new bounding box.
    """
    blobs, unused_im_scale_factors = _get_blobs(im, boxes)
    pred_bbox = np.zeros(boxes.shape, dtype=np.float32)

    if boxes.shape[0] < 1:
        return pred_bbox
        # for j in range(im_rois.shape[0]):
            # box = boxes[i * batch_size + j, :]
            # cv_im = im.copy()
            # cv2.rectangle(cv_im, (box[0], box[1]), (box[2], box[3]), (0, 255, 0))
            # cv2.rectangle(cv_im, (im_rois[j, 0], im_rois[j, 1]), (im_rois[j, 2], im_rois[j, 3]), (0, 0, 255))
            
            # cv2.imshow('test', cv_im)
            # raw_input('Press enter to continue')

    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['rois'].reshape(*(blobs['rois'].shape))
    blobs_out = net.forward(data=blobs['data'].astype(np.float32, copy=False),
                            rois=blobs['rois'].astype(np.float32, copy=False))['bbox_reg']

    im_rois = enlarge_rois(boxes, im.shape[1], im.shape[0])

    widths = im_rois[:, 2:3] - im_rois[:, 0:1] + cfg.EPS
    heights = im_rois[:, 3:4] - im_rois[:, 1:2] + cfg.EPS
    ctr_x = im_rois[:, 0:1] + 0.5 * widths
    ctr_y = im_rois[:, 1:2] + 0.5 * heights

    dx = blobs_out[:, 0:1]
    dy = blobs_out[:, 1:2]
    dw = blobs_out[:, 2:3]
    dh = blobs_out[:, 3:4]

    # pred_ctr_x = dx * widths + ctr_x
    # pred_ctr_y = dy * heights + ctr_y
    pred_ctr_x = dx * widths + im_rois[:, 0:1]
    pred_ctr_y = dy * heights + im_rois[:, 1:2]
    # pred_w = np.exp(dw) * widths
    # pred_h = np.exp(dh) * heights
    pred_w = np.log(dw) * widths
    pred_h = np.log(dh) * heights

    pred_x1 = pred_ctr_x - 0.5 * pred_w
    pred_y1 = pred_ctr_y - 0.5 * pred_h
    pred_x2 = pred_ctr_x + 0.5 * pred_w
    pred_y2 = pred_ctr_y + 0.5 * pred_h

    pred_bbox[:, 0:4] = \
        np.hstack((pred_x1, pred_y1, pred_x2, pred_y2)).astype(np.float32)
    pred_bbox[:, 4] = boxes[:, 4]

    return pred_bbox

def _vis_detection(im, bbox):
    for i in range(bbox.shape[0]):
        box = bbox[i, :]
        cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (0, 255, 0))

    cv2.imshow('test', im)
    raw_input('Press ENTER to continue')

def test_net(net, imdb):
    """
    Test a BBOX network on an image database
    """
    # output_dir = get_output_dir(imdb, net)
    # det_file = os.path.join(output_dir, 'detections.pkl')

    det_file = './output/no_bbox_reg/voc_2007_test/caffenet_fast_rcnn_iter_40000/detections.pkl'
    if not os.path.isfile(det_file):
        print 'Please first obtain detections results without bounding box regression'
        exit(-1)

    with open(det_file, 'rb') as f:
        all_boxes = cPickle.load(f)

    _t = {'bbox_detect': Timer()}

    num_images = len(imdb.image_index)
    for i in xrange(num_images):
        im = cv2.imread(imdb.image_path_at(i))

        _t['bbox_detect'].tic()
        for j in xrange(1, imdb.num_classes):
            # _vis_detection(im.copy(), all_boxes[j][i])
            all_boxes[j][i] = bbox_detect(net, im, all_boxes[j][i])
            # _vis_detection(im.copy(), all_boxes[j][i])
        
        _t['bbox_detect'].toc()
        print 'bbox_detect: {:d}/{:d} {:.3f}s' \
            .format(i + 1, num_images, _t['bbox_detect'].average_time)

    det_file = './output/no_bbox_reg/voc_2007_test/caffenet_fast_rcnn_iter_40000/detections_bbox_exp_hei.pkl'
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)
