import os
import cv2
import time
import h5py
import random
import mcubes
import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from ops import *
from data_utils import *


class IMDecoder(object):
    def __init__(self, sess, real_size, batch_size_input, is_training=False, z_dim=128, ef_dim=32, gf_dim=128, fourier_max_freq=10, checkpoint_dir='./checkpoint/IMDecoder', data_dir='./data', dataset_name='ChairPML'):
        self.sess = sess

        self.real_size = real_size
        self.batch_size_input = batch_size_input
        
        self.batch_size = 16*16*16*4
        if self.batch_size_input<self.batch_size:
            self.batch_size = self.batch_size_input

        self.input_size = 64

        self.z_dim = z_dim
        self.ef_dim = ef_dim
        self.gf_dim = gf_dim

        self.fourier_max_freq = fourier_max_freq

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir

        if is_training:
            phase = 'train'
        else:
            phase = 'test'

        self.n_parts_list = []
        data_voxels = []
        data_points = []
        data_values = []
        parts_info = load_part_data_info(dataset_name, phase)
        curr_shape_path = ''
        for part_info in parts_info:
            shape_path, part_idx = part_info
            n_parts, part_voxel, part_points, part_values = load_from_hdf5_for_IMDecoder(shape_path, part_idx, self.real_size)
            if shape_path != curr_shape_path:
                self.n_parts_list.append(n_parts)
            curr_shape_path = shape_path
            data_voxels.append(part_voxel)
            data_points.append(part_points)
            data_values.append(part_values)
        self.data_voxels = np.expand_dims(np.array(data_voxels), axis=-1)
        self.data_points = np.array(data_points)
        self.data_values = np.array(data_values)

        if os.path.exists(self.data_dir+'/'+self.dataset_name+'_'+phase+'_z.hdf5'):
            data_dict = h5py.File(self.data_dir+'/'+self.dataset_name+'_'+phase+'_z.hdf5', 'r')
            self.z_vectors = data_dict['z'][:]
        else:
            print('Error: cannot load '+self.data_dir+'/'+self.dataset_name+'_'+phase+'_z.hdf5')
            exit(0)
        
        if not is_training:
            self.real_size = 64
            self.test_size = 32
            self.batch_size = self.test_size*self.test_size*self.test_size

            dima = self.test_size
            dim = self.real_size
            self.aux_x = np.zeros([dima,dima,dima],np.uint8)
            self.aux_y = np.zeros([dima,dima,dima],np.uint8)
            self.aux_z = np.zeros([dima,dima,dima],np.uint8)
            multiplier = int(dim/dima)
            multiplier2 = multiplier*multiplier
            multiplier3 = multiplier*multiplier*multiplier
            for i in range(dima):
                for j in range(dima):
                    for k in range(dima):
                        self.aux_x[i,j,k] = i*multiplier
                        self.aux_y[i,j,k] = j*multiplier
                        self.aux_z[i,j,k] = k*multiplier
            self.coords = np.zeros([multiplier3,dima,dima,dima,3],np.float32)
            for i in range(multiplier):
                for j in range(multiplier):
                    for k in range(multiplier):
                        self.coords[i*multiplier2+j*multiplier+k,:,:,:,0] = self.aux_x+i
                        self.coords[i*multiplier2+j*multiplier+k,:,:,:,1] = self.aux_y+j
                        self.coords[i*multiplier2+j*multiplier+k,:,:,:,2] = self.aux_z+k
            self.coords = (self.coords+0.5)/dim*2.0-1.0
            self.coords = np.reshape(self.coords,[multiplier3,self.batch_size,3])

        self.build_model()

    def build_model(self):
        self.z_vector = tf.placeholder(shape=[1,self.z_dim], dtype=tf.float32)
        if self.fourier_max_freq > 0:
            self.point_coord = tf.placeholder(shape=[self.batch_size,3*self.fourier_max_freq], dtype=tf.float32)
        else:
            self.point_coord = tf.placeholder(shape=[self.batch_size,3], dtype=tf.float32)
        self.point_value = tf.placeholder(shape=[self.batch_size,1], dtype=tf.float32)

        self.D = self.im_decoder(self.point_coord, self.z_vector, phase_train=True, reuse=False)
        self.sD = self.im_decoder(self.point_coord, self.z_vector, phase_train=False, reuse=True)
        
        self.loss = tf.reduce_mean(tf.square(self.point_value - self.D))
        
        self.saver = tf.train.Saver(max_to_keep=10)

    def im_decoder(self, points, z, phase_train=True, reuse=False):
        with tf.variable_scope('im_decoder') as scope:
            if reuse:
                scope.reuse_variables()

            zs = tf.tile(z, [self.batch_size,1])
            pointz = tf.concat([points,zs],1)
            print('pointz',pointz.shape)

            im_dec_lin1 = leaky_relu(fc(pointz, self.gf_dim*16, phase_train=phase_train, scope='im_dec_lin1'))
            im_dec_lin1 = tf.concat([im_dec_lin1,pointz],1)

            im_dec_lin2 = leaky_relu(fc(im_dec_lin1, self.gf_dim*8, phase_train=phase_train, scope='im_dec_lin2'))
            im_dec_lin2 = tf.concat([im_dec_lin2,pointz],1)

            im_dec_lin3 = leaky_relu(fc(im_dec_lin2, self.gf_dim*4, phase_train=phase_train, scope='im_dec_lin3'))
            im_dec_lin3 = tf.concat([im_dec_lin3,pointz],1)

            im_dec_lin4 = leaky_relu(fc(im_dec_lin3, self.gf_dim*2, phase_train=phase_train, scope='im_dec_lin4'))
            im_dec_lin4 = tf.concat([im_dec_lin4,pointz],1)

            im_dec_lin5 = leaky_relu(fc(im_dec_lin4, self.gf_dim, phase_train=phase_train, scope='im_dec_lin5'))
            im_dec_lin6 = tf.nn.sigmoid(fc(im_dec_lin5, 1, phase_train=phase_train, scope='im_dec_lin6'))

            return tf.reshape(im_dec_lin6, [self.batch_size,1])

    def get_fourier_features(self, points):
        if self.fourier_max_freq > 0:
            bvals = 2.**np.arange(self.fourier_max_freq/2)
            bvals = np.reshape(np.eye(3)*bvals[:,None,None], [len(bvals)*3,3])
            avals = np.ones((bvals.shape[0])) 
            points_flat = np.reshape(points, [-1,3])

            ff = np.concatenate([avals * np.sin(points_flat @ bvals.T), 
                        avals * np.cos(points_flat @ bvals.T)], axis=-1)
            return ff

        else:
            return points
    
    def train(self, epoch_num, learning_rate, sample_dir='./sample/IMDecoder-samples-train'):
        optim = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        
        batch_idxs = len(self.z_vectors)
        batch_index_list = np.arange(batch_idxs)
        batch_num = int(self.batch_size_input/self.batch_size)
        if self.batch_size_input%self.batch_size != 0:
            print('batch_size_input % batch_size != 0')
            exit(0)

        sample_dir = os.path.join(sample_dir, self.model_dir)
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

        counter = 0
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter+1
            print(' [*] Load SUCCESS')
        else:
            print(' [!] Load failed...')

        avg_loss_list = []
        for epoch in range(counter, epoch_num):
            np.random.shuffle(batch_index_list)
            avg_loss = 0
            avg_num = 0
            for idx in range(0, batch_idxs):
                for minib in range(batch_num):
                    dxb = batch_index_list[idx]
                    batch_z_vector = self.z_vectors[dxb:dxb+1]
                    batch_points_int = np.rint(self.data_points[dxb,minib*self.batch_size:(minib+1)*self.batch_size])
                    batch_points = (batch_points_int+0.5)/self.real_size*2.0-1.0
                    batch_values = self.data_values[dxb,minib*self.batch_size:(minib+1)*self.batch_size]

                    _, loss = self.sess.run([optim, self.loss],
                        feed_dict={
                            self.z_vector: batch_z_vector,
                            self.point_coord: self.get_fourier_features(batch_points),
                            self.point_value: batch_values,
                        })
                    avg_loss += loss
                    avg_num += 1
                    if (idx%16 == 0):
                        print('Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, loss: %.8f, average loss: %.8f' % (epoch, epoch_num, idx, batch_idxs, time.time() - start_time, loss, avg_loss/avg_num))

                if idx==batch_idxs-1:
                    model_float = np.zeros([self.real_size,self.real_size,self.real_size],np.float32)
                    real_model_float = np.zeros([self.real_size,self.real_size,self.real_size],np.float32)
                    for minib in range(batch_num):
                        dxb = batch_index_list[idx]
                        batch_z_vector = self.z_vectors[dxb:dxb+1]
                        batch_points_int = np.rint(self.data_points[dxb,minib*self.batch_size:(minib+1)*self.batch_size]).astype(np.int32)
                        batch_points = (batch_points_int+0.5)/self.real_size*2.0-1.0
                        batch_values = self.data_values[dxb,minib*self.batch_size:(minib+1)*self.batch_size]

                        model_out = self.sess.run(self.sD,
                            feed_dict={
                                self.z_vector: batch_z_vector,
                                self.point_coord: self.get_fourier_features(batch_points),
                            })
                        model_float[batch_points_int[:,0],batch_points_int[:,1],batch_points_int[:,2]] = np.reshape(model_out, [self.batch_size])
                        real_model_float[batch_points_int[:,0],batch_points_int[:,1],batch_points_int[:,2]] = np.reshape(batch_values, [self.batch_size])
                    img1 = np.clip(np.amax(model_float, axis=0)*256, 0, 255).astype(np.uint8)
                    img2 = np.clip(np.amax(model_float, axis=1)*256, 0, 255).astype(np.uint8)
                    img3 = np.clip(np.amax(model_float, axis=2)*256, 0, 255).astype(np.uint8)
                    cv2.imwrite(sample_dir+'/'+str(epoch)+'_1.png', img1)
                    cv2.imwrite(sample_dir+'/'+str(epoch)+'_2.png', img2)
                    cv2.imwrite(sample_dir+'/'+str(epoch)+'_3.png', img3)
                    img1 = np.clip(np.amax(real_model_float, axis=0)*256, 0, 255).astype(np.uint8)
                    img2 = np.clip(np.amax(real_model_float, axis=1)*256, 0, 255).astype(np.uint8)
                    img3 = np.clip(np.amax(real_model_float, axis=2)*256, 0, 255).astype(np.uint8)
                    cv2.imwrite(sample_dir+'/'+str(epoch)+'_1_gt.png', img1)
                    cv2.imwrite(sample_dir+'/'+str(epoch)+'_2_gt.png', img2)
                    cv2.imwrite(sample_dir+'/'+str(epoch)+'_3_gt.png', img3)
                    print('[sample]')

                if idx==batch_idxs-1:
                    self.save(self.checkpoint_dir, epoch)

            avg_loss_list.append(avg_loss/avg_num)

        plt.figure()
        plt.plot(avg_loss_list)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig(os.path.join(sample_dir, 'loss_' + str(self.real_size) + '.png'))
        plt.savefig(os.path.join(sample_dir, 'loss_' + str(self.real_size) + '.pdf'))
        f = open(os.path.join(sample_dir, 'loss_' + str(self.real_size) + '.txt'), 'w')
        for row in avg_loss_list:
            f.write(str(row) + '\n')
        f.close()

    def test(self, sample_dir='./sample/IMDecoder-samples-test'):
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            print(' [*] Load SUCCESS')
        else:
            print(' [!] Load failed...')
            return

        sample_dir = os.path.join(sample_dir, self.model_dir)
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        
        dima = self.test_size
        dim = self.real_size
        multiplier = int(dim/dima)
        multiplier2 = multiplier*multiplier

        for t in range(len(self.z_vectors)):
            model_float = np.zeros([self.real_size+2,self.real_size+2,self.real_size+2],np.float32)
            for i in range(multiplier):
                for j in range(multiplier):
                    for k in range(multiplier):
                        minib = i*multiplier2+j*multiplier+k
                        batch_z = self.z_vectors[t:t+1]
                        model_out = self.sess.run(self.sD,
                            feed_dict={
                                self.z_vector: batch_z,
                                self.point_coord: self.get_fourier_features(self.coords[minib]),
                            })
                        model_float[self.aux_x+i+1,self.aux_y+j+1,self.aux_z+k+1] = np.reshape(model_out, [self.test_size,self.test_size,self.test_size])
            img1 = np.clip(np.amax(model_float, axis=0)*256, 0,255).astype(np.uint8)
            img2 = np.clip(np.amax(model_float, axis=1)*256, 0,255).astype(np.uint8)
            img3 = np.clip(np.amax(model_float, axis=2)*256, 0,255).astype(np.uint8)
            cv2.imwrite(sample_dir+'/'+str(t)+'_1t.png',img1)
            cv2.imwrite(sample_dir+'/'+str(t)+'_2t.png',img2)
            cv2.imwrite(sample_dir+'/'+str(t)+'_3t.png',img3)
            
            thres = 0.5
            vertices, triangles = mcubes.marching_cubes(model_float, thres)
            mcubes.export_mesh(vertices, triangles, sample_dir+'/'+str(t)+'.dae')
        
            print('[sample]')

    def test_z(self, batch_z):
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            print(' [*] Load SUCCESS')
        else:
            print(' [!] Load failed...')
            return
        
        dima = self.test_size
        dim = self.real_size
        multiplier = int(dim/dima)
        multiplier2 = multiplier*multiplier

        samples_list = []
        for t in range(batch_z.shape[0]):
            model_float = np.zeros([self.real_size,self.real_size,self.real_size],np.float32)
            for i in range(multiplier):
                for j in range(multiplier):
                    for k in range(multiplier):
                        minib = i*multiplier2+j*multiplier+k
                        model_out = self.sess.run(self.sD,
                            feed_dict={
                                self.point_coord: self.get_fourier_features(self.coords[minib]),
                                self.z_vector: batch_z[t:t+1],
                            })
                        model_float[self.aux_x+i,self.aux_y+j,self.aux_z+k] = np.reshape(model_out, [self.test_size,self.test_size,self.test_size])
            samples_list.append(model_float)

        return np.stack(samples_list)

    @property
    def model_dir(self):
        return '{}_{}_{}'.format(self.dataset_name, self.input_size, self.fourier_max_freq)
            
    def save(self, checkpoint_dir, step):
        model_name = 'IMDecoder.model'
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(' [*] Reading checkpoints...')
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer('(\d+)(?!.*\d)',ckpt_name)).group(0))
            print(' [*] Success to read {}'.format(ckpt_name))
            return True, counter
        else:
            print(' [*] Failed to find a checkpoint')
            return False, 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='ChairPML')
    parser.add_argument('--train', action='store_true', help='train or test')
    parser.add_argument('--real_size', type=int, default=64)
    parser.add_argument('--batch_size_input', type=int, default=32768)
    parser.add_argument('--epoch_num', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.00005)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    is_train = args.train
    real_size = args.real_size
    batch_size_input = args.batch_size_input
    epoch_num = args.epoch_num
    learning_rate= args.learning_rate

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        imdecoder = IMDecoder(sess=sess, real_size=real_size, batch_size_input=batch_size_input, is_training=is_train, dataset_name=dataset_name)
        if is_train:
            imdecoder.train(epoch_num=epoch_num, learning_rate=learning_rate)
        else:
            imdecoder.test()
