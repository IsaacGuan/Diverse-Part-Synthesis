import os
import json
import h5py
import numpy as np

DATA_DIR = 'data/'
SPLIT_DIR = 'data/train_val_test_split'

def collect_data_id(classname, phase):
    filename = os.path.join(SPLIT_DIR, '{}.{}.json'.format(classname, phase))
    if not os.path.exists(filename):
        raise ValueError('Invalid filepath: {}'.format(filename))

    all_ids = []
    with open(filename, 'r') as fp:
        info = json.load(fp)
    for item in info:
        all_ids.append(item['anno_id'])

    return all_ids

def load_part_data_info(class_name, phase):
    shape_names = collect_data_id(class_name, phase)
    with open('data/{}_info.json'.format(class_name), 'r') as fp:
        nparts_dict = json.load(fp)
    parts_info = []
    for name in shape_names:
        shape_h5_path = os.path.join(DATA_DIR+class_name, name+'.h5')
        if not os.path.exists(shape_h5_path):
            continue
        parts_info.extend([(shape_h5_path, x) for x in range(nparts_dict[name])])

    return parts_info

def load_from_hdf5_for_PCN(path, idx):
    with h5py.File(path, 'r') as data_dict:
        n_parts = data_dict.attrs['n_parts']
        part_voxel = data_dict['parts_voxel_scaled64'][idx].astype(np.float)
        part_voxel_t = data_dict['parts_voxel64'][idx].astype(np.float)
        scale = 1.0 / data_dict['scales'][idx]
        translation_tmp = data_dict['translations'][idx] - (np.array([0,0,0])+np.array([63,63,63]))/2
        translation = -scale*(np.array([translation_tmp[1], translation_tmp[0], translation_tmp[2]])/32)
    return n_parts, part_voxel, part_voxel_t, scale, translation

def load_from_hdf5_for_IMDecoder(path, idx, resolution=64, rescale=False):
    with h5py.File(path, 'r') as data_dict:
        n_parts = data_dict.attrs['n_parts']
        part_voxel = data_dict['parts_voxel_scaled64'][idx].astype(np.float)
        part_points = data_dict['points_{}'.format(resolution)][idx]
        part_values = data_dict['values_{}'.format(resolution)][idx]
    if rescale:
        part_points = part_points / resolution
    return n_parts, part_voxel, part_points, part_values
