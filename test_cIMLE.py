import os
import cv2
import h5py
import time
import mcubes
import random
import argparse
import numpy as np
import tensorflow as tf

from PCN import PCN
from cIMLE import cIMLE
from IMDecoder import IMDecoder
from transformer import *

DATA_DIR = './data'
SAMPLE_DIR = './sample/samples-cIMLE'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='ChairPML')
    args = parser.parse_args()
    dataset_name = args.dataset_name

    data_dict = h5py.File(DATA_DIR+'/'+dataset_name+'_part_assemblies_z.hdf5', 'r')
    part_assemblies_z = data_dict['z'][:]
    shape_id = random.randrange(part_assemblies_z.shape[0])
    data_dict = h5py.File(DATA_DIR+'/'+dataset_name+'_suggestions.hdf5', 'r')
    part_assembly = data_dict['part_assembly'][shape_id:shape_id+1]

    if not os.path.exists(SAMPLE_DIR):
        os.makedirs(SAMPLE_DIR)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        cimle = cIMLE(sess=sess, batch_size=1, noise_num=4, dataset_name=dataset_name)
        ys = cimle.test(part_assemblies_z_test=part_assemblies_z[shape_id:shape_id+1])
    sess.close()

    tf.reset_default_graph()

    with tf.Session(config=run_config) as sess:
        pcn = PCN(sess=sess, batch_size=4, dataset_name=dataset_name)
        _, scales, translations, samples = pcn.test_z(batch_z=ys[0])
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
        parts_float = im_decoder.test_z(ys[0])
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
