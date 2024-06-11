import os
import cv2
import h5py
import mcubes
import random
import argparse
import numpy as np
import tensorflow as tf
import torch

from scipy import spatial

from PCN import PCN
from IMDecoder import IMDecoder
from data_utils import *
from transformer import *

from conditional_denoising_diffusion_pytorch_1d import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D

DATA_DIR = './data'
SAMPLE_DIR = './sample/iterative-generation-cDDPM'

thres = 0.5
n = 100

def generate(diffusion, part_assemblies, iter_num, fourier_max_freq, sample_dir):
    if not os.path.exists(sample_dir+'/iter_'+str(iter_num)):
        os.makedirs(sample_dir+'/iter_'+str(iter_num))

    tf.reset_default_graph()
    with tf.Session(config=run_config) as sess:
        zs = []
        pcn = PCN(sess=sess, batch_size=1, dataset_name=dataset_name)
        could_load, checkpoint_counter = pcn.load(pcn.checkpoint_dir)
        for i in range(part_assemblies.shape[0]):
            z = sess.run(pcn.sE,
                feed_dict={
                    pcn.parts: part_assemblies[i:(i+1)],
                })
            z = np.expand_dims(z, axis=0)
            z = torch.tensor(z)
            condition = torch.repeat_interleave(z, 4, dim=0).cuda()
            sampled_seq = diffusion.sample(condition, batch_size=4)
            sampled_z = sampled_seq.cpu().detach().numpy().squeeze()
            zs.append(sampled_z)
        zs = np.array(zs)
    sess.close()

    scales_list = []
    translations_list = []
    parts_list = []

    tf.reset_default_graph()
    with tf.Session(config=run_config) as sess:
        pcn = PCN(sess=sess, batch_size=4, dataset_name=dataset_name)
        for i in range(part_assemblies.shape[0]):
            if not os.path.exists(sample_dir+'/iter_'+str(iter_num)+'/part_assembly_'+str(i)):
                os.makedirs(sample_dir+'/iter_'+str(iter_num)+'/part_assembly_'+str(i))
            vertices, triangles = mcubes.marching_cubes(part_assemblies[i].squeeze(), thres)
            mcubes.export_mesh(vertices, triangles, sample_dir+'/iter_'+str(iter_num)+'/part_assembly_'+str(i)+'/'+'part_assembly_vox.dae', str(i))
            _, scales, translations, parts = pcn.test_z(batch_z=zs[i])
            scales_list.append(scales)
            translations_list.append(translations)
            parts_list.append(parts)
            for j in range(parts.shape[0]):
                img1 = np.clip(np.amax(parts[j].squeeze(), axis=0)*256, 0,255).astype(np.uint8)
                img2 = np.clip(np.amax(parts[j].squeeze(), axis=1)*256, 0,255).astype(np.uint8)
                img3 = np.clip(np.amax(parts[j].squeeze(), axis=2)*256, 0,255).astype(np.uint8)
                cv2.imwrite(sample_dir+'/iter_'+str(iter_num)+'/part_assembly_'+str(i)+'/'+str(j)+'_1_vox.png',img1)
                cv2.imwrite(sample_dir+'/iter_'+str(iter_num)+'/part_assembly_'+str(i)+'/'+str(j)+'_2_vox.png',img2)
                cv2.imwrite(sample_dir+'/iter_'+str(iter_num)+'/part_assembly_'+str(i)+'/'+str(j)+'_3_vox.png',img3)
                vertices, triangles = mcubes.marching_cubes(parts[j].squeeze(), thres)
                mcubes.export_mesh(vertices, triangles, sample_dir+'/iter_'+str(iter_num)+'/part_assembly_'+str(i)+'/'+str(j)+'_vox.dae', str(j))
    sess.close()

    tf.reset_default_graph()
    with tf.Session(config=run_config) as sess:
        im_decoder = IMDecoder(sess=sess, real_size=64, batch_size_input=32768, fourier_max_freq=fourier_max_freq, dataset_name=dataset_name)
        for i in range(part_assemblies.shape[0]):
            parts_float = im_decoder.test_z(zs[i])
            parts_float_T = transform(parts_float, scales_list[i], translations_list[i]).eval()
            for j in range(parts_float_T.shape[0]):
                img1 = np.clip(np.amax(parts_float_T[j].squeeze(), axis=0)*256, 0,255).astype(np.uint8)
                img2 = np.clip(np.amax(parts_float_T[j].squeeze(), axis=1)*256, 0,255).astype(np.uint8)
                img3 = np.clip(np.amax(parts_float_T[j].squeeze(), axis=2)*256, 0,255).astype(np.uint8)
                cv2.imwrite(sample_dir+'/iter_'+str(iter_num)+'/part_assembly_'+str(i)+'/'+str(j)+'_1.png',img1)
                cv2.imwrite(sample_dir+'/iter_'+str(iter_num)+'/part_assembly_'+str(i)+'/'+str(j)+'_2.png',img2)
                cv2.imwrite(sample_dir+'/iter_'+str(iter_num)+'/part_assembly_'+str(i)+'/'+str(j)+'_3.png',img3)
                vertices, triangles = mcubes.marching_cubes(parts_float_T[j].squeeze(), thres)
                mcubes.export_mesh(vertices, triangles, sample_dir+'/iter_'+str(iter_num)+'/part_assembly_'+str(i)+'/'+str(j)+'.dae', str(j))
    sess.close()

    return parts_list


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

    if dataset_name == 'ChairPML':
        fourier_max_freq = 0
    else:
        fourier_max_freq = 10

    data_parts = []
    data_parts_t = []
    shapes = []
    parts_info = load_part_data_info(dataset_name, 'train')
    curr_shape_path = ''
    for part_info in parts_info:
        shape_path, part_idx = part_info
        n_parts, part_voxel, part_voxel_t, _, _ = load_from_hdf5_for_PCN(shape_path, part_idx)
        if shape_path != curr_shape_path:
            if curr_shape_path != '':
                shapes.append(shape)
            shape = part_voxel_t.copy()
        else:
            shape += part_voxel_t.copy()
        curr_shape_path = shape_path
        data_parts.append(part_voxel)
        data_parts_t.append(part_voxel_t)
    data_parts = np.expand_dims(np.array(data_parts), axis=-1)
    data_parts_t = np.expand_dims(np.array(data_parts_t), axis=-1)
    shapes = np.expand_dims(np.array(shapes), axis=-1)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    part_ids = random.sample(range(0, data_parts_t.shape[0]), n)
    for part_id in part_ids:
        part = np.float32(data_parts_t[part_id:part_id+1])

        if os.path.exists(DATA_DIR+'/'+dataset_name+'_train_z.hdf5'):
            data_dict = h5py.File(DATA_DIR+'/'+dataset_name+'_train_z.hdf5', 'r')
            z_vector = data_dict['z'][part_id:part_id+1]
        else:
            print('Error: cannot load '+DATA_DIR+'/'+dataset_name+'_train_z.hdf5')
            exit(0)

        if not os.path.exists(SAMPLE_DIR+'/'+dataset_name+'/'+str(part_id)):
            os.makedirs(SAMPLE_DIR+'/'+dataset_name+'/'+str(part_id))

        vertices, triangles = mcubes.marching_cubes(part.squeeze(), thres)
        mcubes.export_mesh(vertices, triangles, SAMPLE_DIR+'/'+dataset_name+'/'+str(part_id)+'/'+'part_initial_vox.dae', str(part_id))

        tf.reset_default_graph()
        with tf.Session(config=run_config) as sess:
            im_decoder = IMDecoder(sess=sess, real_size=64, batch_size_input=32768, fourier_max_freq=fourier_max_freq, dataset_name=dataset_name)
            part_float = im_decoder.test_z(z_vector)
        sess.close()

        tf.reset_default_graph()
        with tf.Session(config=run_config) as sess:
            pcn = PCN(sess=sess, batch_size=1, dataset_name=dataset_name)
            _, scale, translation, _ = pcn.test_z(z_vector)
            part_float_T = transform(part_float, scale, translation).eval()
            vertices, triangles = mcubes.marching_cubes(part_float_T.squeeze(), thres)
            mcubes.export_mesh(vertices, triangles, SAMPLE_DIR+'/'+dataset_name+'/'+str(part_id)+'/'+'part_initial.dae', str(part_id))
        sess.close()

        # iter 1
        parts_list = generate(diffusion=diffusion, part_assemblies=part, iter_num=1, fourier_max_freq=fourier_max_freq, sample_dir=SAMPLE_DIR+'/'+dataset_name+'/'+str(part_id))

        part_assembly_list_1 = []
        for i in range(parts_list[0].shape[0]):
            part_assembly = parts_list[0][i] + part
            part_assembly_list_1.append(part_assembly)
        part_assemblies = np.concatenate(part_assembly_list_1)

        # iter 2
        parts_list = generate(diffusion=diffusion, part_assemblies=part_assemblies, iter_num=2, fourier_max_freq=fourier_max_freq, sample_dir=SAMPLE_DIR+'/'+dataset_name+'/'+str(part_id))

        part_assembly_list_2 = []
        for i in range(len(part_assembly_list_1)):
            for j in range(parts_list[i].shape[0]):
                part_assembly = part_assembly_list_1[i] + parts_list[i][j]
                part_assembly_list_2.append(part_assembly)
        part_assemblies = np.concatenate(part_assembly_list_2)

        full_shapes_list_1 = []
        for i in range(len(part_assembly_list_1)):
            full_shape_list = []
            for j in range(parts_list[i].shape[0]):
                full_shape = part_assembly_list_1[i] + parts_list[i][j]
                full_shape_list.append(full_shape)
            full_shapes_list_1.append(full_shape_list)

        for i in range(len(full_shapes_list_1)):
            for j in range(len(full_shapes_list_1[i])):
                np.save(SAMPLE_DIR+'/'+dataset_name+'/'+str(part_id)+'/iter_2/part_assembly_'+str(i)+'/'+str(j)+'_full_shape.npy', full_shapes_list_1[i][j])
                # shapes_flatten = shapes.reshape(*shapes.shape[:-4], -1)
                # shape_flatten = full_shapes_list_1[i][j].reshape(*full_shapes_list_1[i][j].shape[:-4], -1)
                # tree = spatial.KDTree(shapes_flatten)
                # _, idx = tree.query(shape_flatten, 1)
                # # print(shapes[idx].shape)
                # np.save(SAMPLE_DIR+'/'+dataset_name+'/'+str(part_id)+'/iter_2/part_assembly_'+str(i)+'/'+str(j)+'_nearest_neighbor_vox.npy', shapes[idx])
                # vertices, triangles = mcubes.marching_cubes(shapes[idx].squeeze(), thres)
                # mcubes.export_mesh(vertices, triangles, SAMPLE_DIR+'/'+dataset_name+'/'+str(part_id)+'/iter_2/part_assembly_'+str(i)+'/'+str(j)+'_nearest_neighbor_vox.dae', str(j))

        if dataset_name == 'ChairShapeNet' or dataset_name == 'AirplaneShapeNet' or dataset_name == 'ChairPML':
            # iter 3
            parts_list = generate(diffusion=diffusion, part_assemblies=part_assemblies, iter_num=3, fourier_max_freq=fourier_max_freq, sample_dir=SAMPLE_DIR+'/'+dataset_name+'/'+str(part_id))

            full_shapes_list_2 = []
            for i in range(len(part_assembly_list_2)):
                full_shape_list = []
                for j in range(parts_list[i].shape[0]):
                    full_shape = part_assembly_list_2[i] + parts_list[i][j]
                    full_shape_list.append(full_shape)
                full_shapes_list_2.append(full_shape_list)

            for i in range(len(full_shapes_list_2)):
                for j in range(len(full_shapes_list_2[i])):
                    np.save(SAMPLE_DIR+'/'+dataset_name+'/'+str(part_id)+'/iter_3/part_assembly_'+str(i)+'/'+str(j)+'_full_shape.npy', full_shapes_list_2[i][j])
                    # shapes_flatten = shapes.reshape(*shapes.shape[:-4], -1)
                    # shape_flatten = full_shapes_list_2[i][j].reshape(*full_shapes_list_2[i][j].shape[:-4], -1)
                    # tree = spatial.KDTree(shapes_flatten)
                    # _, idx = tree.query(shape_flatten, 1)
                    # # print(shapes[idx].shape)
                    # np.save(SAMPLE_DIR+'/'+dataset_name+'/'+str(part_id)+'/iter_3/part_assembly_'+str(i)+'/'+str(j)+'_nearest_neighbor_vox.npy', shapes[idx])
                    # vertices, triangles = mcubes.marching_cubes(shapes[idx].squeeze(), thres)
                    # mcubes.export_mesh(vertices, triangles, SAMPLE_DIR+'/'+dataset_name+'/'+str(part_id)+'/iter_3/part_assembly_'+str(i)+'/'+str(j)+'_nearest_neighbor_vox.dae', str(j))
