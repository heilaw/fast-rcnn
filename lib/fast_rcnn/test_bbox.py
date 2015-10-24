from fast_rcnn.config import cfg, get_output_dir
from utils.blob import im_list_to_blob
from utils.timer import Timer

import cv2
import os.path
import numpy as np
import cPickle

def _enlarge_boxes(boxes, im_width, im_height):
    w = (boxes[:, 2] - boxes[:, 0] + 1)[:, np.newaxis]
    h = (boxes[:, 3] - boxes[:, 1] + 1)[:, np.newaxis]
    r = (w + h) / 4

    x_c = np.floor((boxes[:, 2] + boxes[:, 0]) / 2)[:, np.newaxis]
    y_c = np.floor((boxes[:, 3] + boxes[:, 1]) / 2)[:, np.newaxis]

    e_boxes = np.hstack((np.maximum(x_c - w / 2 - r, 1),
        np.maximum(y_c - h / 2 - r, 1),
        np.minimum(x_c + w / 2 + r, im_width),
        np.minimum(y_c + h / 2 + r, im_height)))
    
    return e_boxes

def _crop_images(im, roi):
    return im[roi[1]:roi[3], roi[0]:roi[2], :]

def _get_blobs(im, rois):
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    processed_ims = [_crop_images(im_orig.copy(), rois[i, :]) for i in xrange(rois.shape[0])]
    im_blob = im_list_to_blob(processed_ims)

    rois = rois - np.hstack((rois[:, 0:1], rois[:, 1:2], rois[:, 0:1], rois[:, 1:2]))

    batch_ind = np.array(range(len(processed_ims)))
    rois_blob = np.hstack((batch_ind[:, np.newaxis], rois))

    return im_blob, rois_blob

def bbox_detect(net, im, boxes):
    """
    Generate new bounding box.
    """
    pred_bbox = np.zeros(boxes.shape, dtype=np.float32)

    batch_size = 32
    for i in range(int(np.ceil(boxes.shape[0] / float(batch_size)))):
        im_rois = _enlarge_boxes(boxes[i * batch_size:min((i + 1) * batch_size, boxes.shape[0]), :], im.shape[1], im.shape[0])

        blobs = {'data': None, 'rois': None}
        blobs['data'], blobs['rois'] = _get_blobs(im, im_rois[:, 0:4].astype(np.float32, copy=True))

        net.blobs['data'].reshape(*(blobs['data'].shape))
        net.blobs['rois'].reshape(*(blobs['rois'].shape))
        blobs_out = net.forward(data=blobs['data'].astype(np.float32, copy=False),
                                rois=blobs['rois'].astype(np.float32, copy=False))['bbox_reg']

        im_width = im_rois[:, 2:3] - im_rois[:, 0:1]
        im_height = im_rois[:, 3:4] - im_rois[:, 1:2]

        im_x = im_width * blobs_out[:, 0:1]
        im_y = im_height * blobs_out[:, 1:2]

        im_x1 = im_x - im_width * blobs_out[:, 2:3] / 2 + im_rois[:, 0:1]
        im_y1 = im_y - im_height * blobs_out[:, 3:4] / 2 + im_rois[:, 1:2]
        im_x2 = im_x + im_width * blobs_out[:, 2:3] / 2 + im_rois[:, 0:1]
        im_y2 = im_y + im_height * blobs_out[:, 3:4] / 2 + im_rois[:, 1:2]

        im_boxes = np.hstack((im_x1, im_y1, im_x2, im_y2)).astype(np.float32)
        pred_bbox[i * batch_size:min((i + 1) * batch_size, boxes.shape[0]), 0:4] = im_boxes
        pred_bbox[i * batch_size:min((i + 1) * batch_size, boxes.shape[0]), 4] = \
            boxes[i * batch_size:min((i + 1) * batch_size, boxes.shape[0]), 4]

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

    det_file = './output/no_bbox_reg/voc_2007_test/caffenet_fast_rcnn_iter_40000/detections_bbox.pkl'
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)
