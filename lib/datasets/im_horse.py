# --------------------------------------------------------
# Written by Yu-Wei Chao
# --------------------------------------------------------

import datasets
# import datasets.pascal_voc
import datasets.im_horse
import os
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess

import os.path as osp

class im_horse(datasets.imdb):
    def __init__(self, image_set, root_dir):
        if (image_set == 'train2015_single' or image_set =='train2015_sigmoid'):
            image_set_org = 'train2015'
        else:
            image_set_org = image_set
        datasets.imdb.__init__(self, 'im_horse_' + image_set, True)
        self._image_set = image_set
        # Set cache root
        self._cache_root = osp.abspath(osp.join(root_dir, 'data', 'cache'))
        if not os.path.exists(self._cache_root):
            os.makedirs(self._cache_root)
        # Set input paths and files
        self._data_path = './caches/im_base/horse/' + image_set_org
        self._det_path = './caches/det_base/horse/' + image_set_org
        self._anno_file = './caches/anno_base/anno_horse.mat'
        self._classes = ('feed', 
                         'groom',
                         'hold',
                         'hug',
                         'jump',
                         'kiss',
                         'load',
                         'hop_on',
                         'pet',
                         'race',
                         'ride',
                         'run',
                         'straddle',
                         'train',
                         'walk',
                         'wash',
                         'no_interaction')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        # Opposed to voc, _image_index here contains the file names
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.object_detection_roidb

        # # PASCAL specific config options
        # self.config = {'cleanup'  : True,
        #                'use_salt' : True,
        #                'top_k'    : 2000}

        # assert os.path.exists(self._devkit_path), \
        #         'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        # image_index contains the file names
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # index contains the file names
        image_path = os.path.join(self._data_path, index)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the file names by listdir. Should change to use selective search
        caches later.
        """
        # image_index contains the file names
        image_index = [f for f in os.listdir(self._data_path) 
                       if os.path.isfile(os.path.join(self._data_path, f))]
        image_index.sort()
        return image_index

    def object_detection_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self._cache_root,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            if self._image_set == 'train2015_single':
                self.update_image_set_index(roidb)  # TODO: drop this line later
            return roidb

        # Load selective search results: no longer needs this
        # roidb = self._load_selective_search_roidb()

        # Load detection bbox and scores
        roidb = self._load_detection_roidb()

        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    # def _load_selective_search_roidb(self):
    #     filename = os.path.abspath(os.path.join(self._data_path, '..', '..',
    #                                             self.name + '.mat'))
    #     assert os.path.exists(filename), \
    #            'Selective search data not found at: {}'.format(filename)
    #     raw_data = sio.loadmat(filename)['boxes'].ravel()
    # 
    #     box_list = []
    #     for i in xrange(raw_data.shape[0]):
    #         box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)
    # 
    #     roidb = self.create_roidb_from_box_list(box_list, None)
    #     # roidb[0].pop("gt_overlaps",None)
    #     # roidb[0].pop("gt_classes",None)
    # 
    #     return roidb

    def _load_detection_roidb(self):
        # Load HICO annotation
        assert(self._image_set == 'train2015_single' or
               self._image_set == 'train2015_sigmoid' or
               self._image_set == 'train2015' or 
               self._image_set == 'test2015')
        if (self._image_set == 'train2015_single' or
            self._image_set == 'train2015' or
            self._image_set == 'train2015_sigmoid'):
            anno = sio.loadmat(self._anno_file)['anno_train']
            lsim = sio.loadmat(self._anno_file)['list_train']
        if self._image_set == 'test2015':
            anno = sio.loadmat(self._anno_file)['anno_test']
            lsim = sio.loadmat(self._anno_file)['list_test']
        
        # Load detection results
        raw_single = [self._load_detection(index, anno, lsim) 
                      for index in self._image_index]
        # Expand the dataset; TODO: drop this later
        if self._image_set == 'train2015_single':
            print 'expanding data for single label training ...'
            for idx, index in enumerate(self._image_index):
                for j in xrange(raw_single[idx]['label'].size):
                    roidb.append({'index' : index,
                                  'boxes' : raw_single[idx]['boxes'], 
                                  'label' : np.array([raw_single[idx]['label'][j]]),
                                  'flipped' : raw_single[idx]['flipped']})
            self.update_image_set_index(roidb)
        if (self._image_set == 'train2015' or 
            self._image_set == 'test2015' or
            self._image_set == 'train2015_sigmoid'):
            roidb = raw_single
            # for rois in roidb:
            #     rois.pop("index", None)
            #     rois.pop("label", None)

        return roidb

    def _load_detection(self, index, anno, lsim):
        # ------- params to be changed later ----------------------------------
        nid = 18  # horse
        # ---------------------------------------------------------------------
         
        # Load detection file
        print index
        filename = os.path.join(self._det_path, os.path.splitext(index)[0] + '.mat')
        #print 'Loading: {}'.format(filename)
        assert os.path.exists(filename), \
               'Detection file not found at: {}'.format(filename)
        res = sio.loadmat(filename)['res']
        dets = res['dets'][0,0][0,nid]
        # NMS: 'keep' will also sort the dets by detection scores
        keep = np.squeeze(res['keep'][0,0][0,nid])
        dets = dets[keep,:]
        # Keep all the detection boxes now and filter later in data fetching
        boxes = dets[:,0:4]
        boxes = np.around(boxes).astype('uint16')        
        # Get image index
        ind = [idx for idx, im in enumerate(lsim) if str(im[0][0]) == index]
        assert(len(ind) == 1)
        
        # Read labels
        if self._image_set == 'train2015_sigmoid':
            labels = anno[:, ind]
            labels[labels != 1] = 0
        else:
            labels = np.where(anno[:,ind] == 1)[0]
        
        return {'boxes' : boxes, 'label' : labels, 'flipped' : False}

    # TODO: dropped this later
    def update_image_set_index(self, roidb):
        image_index = []
        for rois in roidb:
            image_index.append(rois['index'])
        self._image_index = image_index

    # TODO: edit this function if neccessary
    def _write_voc_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        comp_id = 'comp4'
        if use_salt:
            comp_id += '-{}'.format(os.getpid())

        # VOCdevkit/results/VOC2007/Main/comp4-44503_det_test_aeroplane.txt
        path = os.path.join(self._devkit_path, 'results', 'VOC' + self._year,
                            'Main', comp_id + '_')
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = path + 'det_' + self._image_set + '_' + cls + '.txt'
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
        return comp_id

    # TODO: edit this function if neccessary
    def _do_matlab_eval(self, comp_id, output_dir='output'):
        rm_results = self.config['cleanup']

        path = os.path.join(os.path.dirname(__file__),
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
               .format(self._devkit_path, comp_id,
                       self._image_set, output_dir, int(rm_results))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    # TODO: edit this function if neccessary
    def evaluate_detections(self, all_boxes, output_dir):
        comp_id = self._write_voc_results_file(all_boxes)
        self._do_matlab_eval(comp_id, output_dir)

    # # TODO: edit this function if neccessary
    # def competition_mode(self, on):
    #     if on:
    #         self.config['use_salt'] = False
    #         self.config['cleanup'] = False
    #     else:
    #         self.config['use_salt'] = True
    #         self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.im_horse('train2015')
    res = d.roidb
    from IPython import embed; embed()
