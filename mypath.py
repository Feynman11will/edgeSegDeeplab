'''
@Author: your name
@Date: 2019-12-30 22:59:50
@LastEditTime : 2019-12-30 23:03:03
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \pytorch-deeplab-xception-master\mypath.py
'''
class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            
            return 'G:/10Coco-voc-dataset--cifar10/VOCdataset/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
