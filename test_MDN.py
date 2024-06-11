import os
import cv2
import h5py
import heapq
import mcubes
import random
import argparse
import numpy as np
import tensorflow as tf

from PCN import PCN
from MDN import MDN
from IMDecoder import IMDecoder
from transformer import *

DATA_DIR = './data'
SAMPLE_DIR = './sample/samples-MDN'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='ChairPML')
    parser.add_argument('--sample_num', type=int, default=100)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    sample_num = args.sample_num

    data_dict = h5py.File(DATA_DIR+'/'+dataset_name+'_part_assemblies_z.hdf5', 'r')
    part_assemblies_z = data_dict['z'][:]
    part_assemblies_z = np.expand_dims(part_assemblies_z, axis=1)
    shape_id = random.randrange(part_assemblies_z.shape[0])
    data_dict = h5py.File(DATA_DIR+'/'+dataset_name+'_suggestions.hdf5', 'r')
    part_assembly = data_dict['part_assembly'][shape_id:shape_id+1]

    if not os.path.exists(SAMPLE_DIR):
        os.makedirs(SAMPLE_DIR)

    mdn = MDN(dataset_name=dataset_name)
    alpha, mu, sigma = mdn.test(part_assemblies_z_test=part_assemblies_z[shape_id:shape_id+1])

    final_mu = []
    final_sigma = []
    for c in range(mdn.cat_num):
        mu_c = []
        sigma_c = []
        for i in range(alpha.shape[0]):
            idx_sorted = heapq.nlargest(mdn.cat_num, range(len(alpha[i, 0])), key=alpha[i, 0].__getitem__)
            alpha_sorted = heapq.nlargest(mdn.cat_num, alpha[i, 0])
            idx_selected = idx_sorted[c]
            alpha_selected = alpha_sorted[c]
            mu_c.append(mu[i, :, idx_selected])
            sigma_c.append(sigma[i, :, idx_selected])
        final_mu.append(mu_c)
        final_sigma.append(sigma_c)
    final_mu = np.array(final_mu)
    final_sigma = np.array(final_sigma)
    final_mu = np.transpose(final_mu, (1, 0, 2))[0]
    final_sigma = np.transpose(final_sigma, (1, 0, 2))[0]

    batch_z_list = []
    for c in range(final_mu.shape[0]):
        if not os.path.exists(SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id)+'/'+str(c)):
            os.makedirs(SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id)+'/'+str(c))
        batch_z = np.random.normal(final_mu[c:c+1], final_sigma[c:c+1], (sample_num, 1, final_mu.shape[1]))
        batch_z = batch_z.reshape((-1, final_mu.shape[1]))
        batch_z_list.append(batch_z)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    tf.reset_default_graph()

    with tf.Session(config=run_config) as sess:
        pcn = PCN(sess=sess, batch_size=sample_num, dataset_name=dataset_name)

        for c in range(final_mu.shape[0]):
            if not os.path.exists(SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id)+'/'+str(c)):
                os.makedirs(SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id)+'/'+str(c))

            batch_z = batch_z_list[c]

            parts, scales, translations, parts_T = pcn.test_z(batch_z)

            for i in range(parts_T.shape[0]):
                img1 = np.clip(np.amax(parts_T[i].squeeze(), axis=0)*256, 0,255).astype(np.uint8)
                img2 = np.clip(np.amax(parts_T[i].squeeze(), axis=1)*256, 0,255).astype(np.uint8)
                img3 = np.clip(np.amax(parts_T[i].squeeze(), axis=2)*256, 0,255).astype(np.uint8)
                cv2.imwrite(SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id)+'/'+str(c)+'/'+str(i)+'_1_vox.png',img1)
                cv2.imwrite(SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id)+'/'+str(c)+'/'+str(i)+'_2_vox.png',img2)
                cv2.imwrite(SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id)+'/'+str(c)+'/'+str(i)+'_3_vox.png',img3)
                np.save(SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id)+'/'+str(c)+'/'+str(i)+'_vox.npy', parts_T[i].squeeze())
                thres = 0.5
                vertices, triangles = mcubes.marching_cubes(parts_T[i].squeeze(), thres)
                mcubes.export_mesh(vertices, triangles, SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id)+'/'+str(c)+'/'+str(i)+'_vox.dae', str(i))
    sess.close()

    tf.reset_default_graph()

    with tf.Session(config=run_config) as sess:
        im_decoder = IMDecoder(sess=sess, real_size=64, batch_size_input=32768, dataset_name=dataset_name)

        for c in range(final_mu.shape[0]):
            if not os.path.exists(SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id)+'/'+str(c)):
                os.makedirs(SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id)+'/'+str(c))

            batch_z = batch_z_list[c]

            parts_float = im_decoder.test_z(batch_z)
            parts_float_T = transform(parts_float, scales, translations).eval()

            for i in range(parts_float_T.shape[0]):
                img1 = np.clip(np.amax(parts_float_T[i].squeeze(), axis=0)*256, 0,255).astype(np.uint8)
                img2 = np.clip(np.amax(parts_float_T[i].squeeze(), axis=1)*256, 0,255).astype(np.uint8)
                img3 = np.clip(np.amax(parts_float_T[i].squeeze(), axis=2)*256, 0,255).astype(np.uint8)
                cv2.imwrite(SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id)+'/'+str(c)+'/'+str(i)+'_1.png',img1)
                cv2.imwrite(SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id)+'/'+str(c)+'/'+str(i)+'_2.png',img2)
                cv2.imwrite(SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id)+'/'+str(c)+'/'+str(i)+'_3.png',img3)
                np.save(SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id)+'/'+str(c)+'/'+str(i)+'.npy', parts_float_T[i].squeeze())
                thres = 0.5
                vertices, triangles = mcubes.marching_cubes(parts_float_T[i].squeeze(), thres)
                mcubes.export_mesh(vertices, triangles, SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id)+'/'+str(c)+'/'+str(i)+'.dae', str(i))
    sess.close()

    vertices, triangles = mcubes.marching_cubes(part_assembly[0].squeeze(), thres)
    mcubes.export_mesh(vertices, triangles, SAMPLE_DIR+'/'+dataset_name+'/'+str(shape_id)+'/'+'part_assembly.dae', str(i))
