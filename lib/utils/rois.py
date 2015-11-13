import numpy as np
from fast_rcnn.config import cfg

def enlarge_rois(rois, im_width, im_height): 
    w = (rois[:, 2] - rois[:, 0] + 1)[:, np.newaxis]
    h = (rois[:, 3] - rois[:, 1] + 1)[:, np.newaxis]
    # r = (w + h) / 4
    w = cfg.TRAIN.EXTRA_WIDTH * w
    h = cfg.TRAIN.EXTRA_HEIGHT * h

    x_c = np.floor((rois[:, 2] + rois[:, 0]) / 2)[:, np.newaxis]
    y_c = np.floor((rois[:, 3] + rois[:, 1]) / 2)[:, np.newaxis]

    e_rois = np.hstack((np.maximum(x_c - w / 2, 0),
        np.maximum(y_c - h / 2, 0),
        np.minimum(x_c + w / 2, im_width - 1),
        np.minimum(y_c + h / 2, im_height - 1)))
    
    return e_rois
