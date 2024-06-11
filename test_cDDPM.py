import os
import cv2
import h5py
import time
import mcubes
import random
import argparse
import numpy as np
import tensorflow as tf
import torch

from conditional_denoising_diffusion_pytorch_1d import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D

from PCN import PCN
from IMDecoder import IMDecoder
from transformer import *
from data_utils import *

DATA_DIR = './data'
SAMPLE_DIR = './sample/samples-cDDPM'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='ChairPML')
    args = parser.parse_args()
    dataset_name = args.dataset_name

    model = Unet1D(
        dim = 64,
        channels = 1
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = 128,
        timesteps = 1000,
        objective = 'pred_v'
    )

    suggestions_data_dict = h5py.File(DATA_DIR+'/'+dataset_name+'_suggestions_z.hdf5', 'r')
    suggestions_z = suggestions_data_dict['z']
    part_assemblies_data_dict = h5py.File(DATA_DIR+'/'+dataset_name+'_part_assemblies_z.hdf5', 'r')
    part_assemblies_z = part_assemblies_data_dict['z']

    suggestions_z_training_seq = torch.Tensor(np.expand_dims(suggestions_z, axis=1))
    part_assemblies_z_training_seq = torch.Tensor(np.expand_dims(part_assemblies_z, axis=1))
    training_seq = torch.cat((suggestions_z_training_seq, part_assemblies_z_training_seq), dim = 1)
    dataset = Dataset1D(training_seq)

    trainer = Trainer1D(
        diffusion,
        dataset = dataset,
        train_batch_size = 32,
        train_lr = 8e-5,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        ema_decay = 0.995,
        amp = True,
        results_folder = './checkpoint/cDDPM/'+dataset_name
    )

    trainer.load('500')

    if not os.path.exists(SAMPLE_DIR):
        os.makedirs(SAMPLE_DIR)

    n_parts_list = []
    data_parts_t = []
    parts_info = load_part_data_info(dataset_name, 'train')
    curr_shape_path = ''
    for part_info in parts_info:
        shape_path, part_idx = part_info
        n_parts, part_voxel, part_voxel_t, scale, translation = load_from_hdf5_for_PCN(shape_path, part_idx)
        if shape_path != curr_shape_path:
            n_parts_list.append(n_parts)
        curr_shape_path = shape_path
        data_parts_t.append(part_voxel_t)
    data_parts_t = np.expand_dims(np.array(data_parts_t), axis=-1)

    shape_id = 70
    data_dict = h5py.File(DATA_DIR+'/'+dataset_name+'_suggestions.hdf5', 'r')
    part_assembly = data_parts_t[shape_id:shape_id+1]

    condition = torch.repeat_interleave(part_assemblies_z_training_seq[shape_id:shape_id+1], 4, dim=0).cuda()
    start_time = time.time()
    sampled_seq = diffusion.sample(condition, batch_size=4)
    print('Diffusion', time.time() - start_time)
    sampled_z = sampled_seq.cpu().detach().numpy().squeeze()

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        pcn = PCN(sess=sess, batch_size=4, dataset_name=dataset_name)
        _, scales, translations, samples = pcn.test_z(batch_z=sampled_z)
        if not os.path.exists(SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id)):
            os.makedirs(SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id))
        for i in range(samples.shape[0]):
            img1 = np.clip(np.amax(samples[i].squeeze(), axis=0)*256, 0,255).astype(np.uint8)
            img2 = np.clip(np.amax(samples[i].squeeze(), axis=1)*256, 0,255).astype(np.uint8)
            img3 = np.clip(np.amax(samples[i].squeeze(), axis=2)*256, 0,255).astype(np.uint8)
            cv2.imwrite(SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id)+'/'+str(i)+'_1_vox.png',img1)
            cv2.imwrite(SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id)+'/'+str(i)+'_2_vox.png',img2)
            cv2.imwrite(SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id)+'/'+str(i)+'_3_vox.png',img3)
            np.save(SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id)+'/'+str(i)+'_vox.npy', samples[i].squeeze())
            thres = 0.5
            vertices, triangles = mcubes.marching_cubes(samples[i].squeeze(), thres)
            mcubes.export_mesh(vertices, triangles, SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id)+'/'+str(i)+'_vox.dae', str(i))
    sess.close()

    tf.reset_default_graph()

    with tf.Session(config=run_config) as sess:
        im_decoder = IMDecoder(sess=sess, real_size=64, batch_size_input=32768, dataset_name=dataset_name)

        if not os.path.exists(SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id)):
            os.makedirs(SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id))

        start_time = time.time()
        parts_float = im_decoder.test_z(sampled_z)
        print('IMDecoder', time.time() - start_time)
        start_time = time.time()
        parts_float_T = transform(parts_float, scales, translations).eval()
        print('transform', time.time() - start_time)

        for i in range(parts_float_T.shape[0]):
            img1 = np.clip(np.amax(parts_float_T[i].squeeze(), axis=0)*256, 0,255).astype(np.uint8)
            img2 = np.clip(np.amax(parts_float_T[i].squeeze(), axis=1)*256, 0,255).astype(np.uint8)
            img3 = np.clip(np.amax(parts_float_T[i].squeeze(), axis=2)*256, 0,255).astype(np.uint8)
            cv2.imwrite(SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id)+'/'+str(i)+'_1.png',img1)
            cv2.imwrite(SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id)+'/'+str(i)+'_2.png',img2)
            cv2.imwrite(SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id)+'/'+str(i)+'_3.png',img3)
            np.save(SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id)+'/'+str(i)+'.npy', parts_float_T[i].squeeze())
            thres = 0.5
            start_time = time.time()
            vertices, triangles = mcubes.marching_cubes(parts_float_T[i].squeeze(), thres)
            mcubes.export_mesh(vertices, triangles, SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id)+'/'+str(i)+'.dae', str(i))
            print('Marching cubes', time.time() - start_time)
    sess.close()

    vertices, triangles = mcubes.marching_cubes(part_assembly[0].squeeze(), thres)
    mcubes.export_mesh(vertices, triangles, SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id)+'/'+'part_assembly.dae', str(i))
