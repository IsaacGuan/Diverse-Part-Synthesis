import argparse

import numpy as np
import tensorflow as tf

from PCN import PCN
from data_utils import *

DATA_DIR = './data'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='ChairPML')
    parser.add_argument('--train', action='store_true', help='train or test')
    args = parser.parse_args()
    dataset_name = args.dataset_name
    is_train = args.train

    if is_train:
        phase = 'train'
    else:
        phase = 'test'

    n_parts_list = []
    parts = []
    parts_info = load_part_data_info(dataset_name, phase)
    curr_shape_path = ''
    for part_info in parts_info:
        shape_path, part_idx = part_info
        n_parts, part_voxel, _, _, _ = load_from_hdf5_for_PCN(shape_path, part_idx)
        if shape_path != curr_shape_path:
            n_parts_list.append(n_parts)
        curr_shape_path = shape_path
        parts.append(part_voxel)
    parts = np.expand_dims(np.array(parts), axis=-1)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        pcn = PCN(sess=sess, batch_size=1, dataset_name=dataset_name)
        if is_train:
            pcn.get_z(batch_parts=parts, dataset_name=dataset_name+'_train')
        else:
            pcn.get_z(batch_parts=parts, dataset_name=dataset_name+'_test')
