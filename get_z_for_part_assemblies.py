import os
import h5py
import argparse

import numpy as np
import tensorflow as tf

from PCN import PCN

DATA_DIR = './data'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='ChairPML')
    args = parser.parse_args()
    dataset_name = args.dataset_name

    if os.path.exists(DATA_DIR+'/'+dataset_name+'_suggestions.hdf5'):
        data_dict = h5py.File(DATA_DIR+'/'+dataset_name+'_suggestions.hdf5', 'r')
        part_assemblies = data_dict['part_assembly'][:]

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        pcn = PCN(sess=sess, batch_size=64, dataset_name=dataset_name)
        pcn.get_z(batch_parts=part_assemblies, dataset_name=dataset_name+'_part_assemblies')
