import os
import h5py
import random
import argparse

import numpy as np

from data_utils import *

DATA_DIR = './data'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='ChairPML')
    parser.add_argument('--loop_num', type=int, default=1024)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    loop_num = args.loop_num

    parts_info = load_part_data_info(dataset_name, 'train')
    n_parts_list = []
    parts = []
    parts_t = []
    curr_shape_path = ''
    for part_info in parts_info:
        shape_path, part_idx = part_info
        n_parts, part_voxel, part_voxel_t, scale, translation = load_from_hdf5_for_PCN(shape_path, part_idx)
        if shape_path != curr_shape_path:
            n_parts_list.append(n_parts)
        curr_shape_path = shape_path
        parts.append(part_voxel)
        parts_t.append(part_voxel_t)

    shapes_w_parts = []
    shapes_w_parts_t = []
    curr_part_num = 0
    for i in range(len(n_parts_list)):
        curr_parts = []
        curr_parts_t = []
        for j in range(n_parts_list[i]):
            curr_parts.append(parts[curr_part_num+j:curr_part_num+j+1])
            curr_parts_t.append(parts_t[curr_part_num+j:curr_part_num+j+1])
        curr_part_num += n_parts_list[i]
        shapes_w_parts.append(np.array(curr_parts))
        shapes_w_parts_t.append(np.array(curr_parts_t))

    hdf5_path = DATA_DIR+'/'+dataset_name+'_suggestions.hdf5'
    hdf5_file = h5py.File(hdf5_path, 'w')

    if dataset_name == 'ChairShapeNet' or dataset_name == 'AirplaneShapeNet' or dataset_name == 'ChairPML':
        hdf5_file.create_dataset('part_assembly', [loop_num*3,64,64,64,1], np.int8, compression=9)
        hdf5_file.create_dataset('suggestion', [loop_num*3,64,64,64,1], np.int8, compression=9)
    else:
        hdf5_file.create_dataset('part_assembly', [loop_num*2,64,64,64,1], np.int8, compression=9)
        hdf5_file.create_dataset('suggestion', [loop_num*2,64,64,64,1], np.int8, compression=9)

    part_assembly_list = []
    suggestion_list = []
    counter = 0

    # 1-component pairs
    for i in range(loop_num):
        print(counter)
        no_enough_parts = True
        while no_enough_parts:
            no_enough_parts = False
            shape_id = random.randint(0,len(n_parts_list)-1)
            if shapes_w_parts_t[shape_id].shape[0] < 2:
                no_enough_parts = True
        part_assembly_idx = np.random.choice(np.arange(shapes_w_parts_t[shape_id].shape[0]), size=1)
        suggestion_idx = np.random.choice(np.setdiff1d(np.arange(shapes_w_parts_t[shape_id].shape[0]), part_assembly_idx), size=1)
        part_assembly = shapes_w_parts_t[shape_id][part_assembly_idx].squeeze()
        suggestion = shapes_w_parts[shape_id][suggestion_idx].squeeze()
        part_assembly_list.append(np.reshape(part_assembly, (64,64,64,1)))
        suggestion_list.append(np.reshape(suggestion, (64,64,64,1)))
        counter += 1

    # 2-component pairs
    for i in range(loop_num):
        print(counter)
        no_enough_parts = True
        while no_enough_parts:
            no_enough_parts = False
            shape_id = random.randint(0,len(n_parts_list)-1)
            if shapes_w_parts_t[shape_id].shape[0] < 3:
                no_enough_parts = True
        part_assembly_idx = np.random.choice(np.arange(shapes_w_parts_t[shape_id].shape[0]), size=2)
        suggestion_idx = np.random.choice(np.setdiff1d(np.arange(shapes_w_parts_t[shape_id].shape[0]), part_assembly_idx), size=1)
        part_assembly = shapes_w_parts_t[shape_id][part_assembly_idx][0].squeeze() + shapes_w_parts_t[shape_id][part_assembly_idx][1].squeeze()
        suggestion = shapes_w_parts[shape_id][suggestion_idx].squeeze()
        part_assembly_list.append(np.reshape(part_assembly, (64,64,64,1)))
        suggestion_list.append(np.reshape(suggestion, (64,64,64,1)))
        counter += 1

    if dataset_name == 'ChairShapeNet' or dataset_name == 'AirplaneShapeNet' or dataset_name == 'ChairPML':
        # 3-component pairs
        for i in range(loop_num):
            print(counter)
            no_enough_parts = True
            while no_enough_parts:
                no_enough_parts = False
                shape_id = random.randint(0,len(n_parts_list)-1)
                if shapes_w_parts_t[shape_id].shape[0] < 4:
                    no_enough_parts = True
            part_assembly_idx = np.random.choice(np.arange(shapes_w_parts_t[shape_id].shape[0]), size=3)
            suggestion_idx = np.random.choice(np.setdiff1d(np.arange(shapes_w_parts_t[shape_id].shape[0]), part_assembly_idx), size=1)
            part_assembly = shapes_w_parts_t[shape_id][part_assembly_idx][0].squeeze() + shapes_w_parts_t[shape_id][part_assembly_idx][1].squeeze() + shapes_w_parts_t[shape_id][part_assembly_idx][2].squeeze()
            suggestion = shapes_w_parts[shape_id][suggestion_idx].squeeze()
            part_assembly_list.append(np.reshape(part_assembly, (64,64,64,1)))
            suggestion_list.append(np.reshape(suggestion, (64,64,64,1)))
            counter += 1

    temp = list(zip(part_assembly_list, suggestion_list))
    random.shuffle(temp)
    part_assembly_list, suggestion_list = zip(*temp)
    part_assembly_list, suggestion_list = list(part_assembly_list), list(suggestion_list)

    hdf5_file['part_assembly'][:] = np.array(part_assembly_list)
    hdf5_file['suggestion'][:] = np.array(suggestion_list)

    hdf5_file.close()
