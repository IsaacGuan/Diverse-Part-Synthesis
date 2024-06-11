import os
import cv2
import h5py
import time
import mcubes
import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from ops import *
from sklearn.utils import shuffle


class cIMLE(object):
    def __init__(self, sess, batch_size, noise_num, z_dim=128, f_dim=128, checkpoint_dir='./checkpoint/cIMLE', data_dir='./data', dataset_name='ChairPML'):
        self.sess = sess

        self.input_size = 64
        self.noise_num = noise_num
        self.batch_size = batch_size

        self.z_dim = z_dim
        self.f_dim = f_dim

        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir
        self.dataset_name = dataset_name

        if os.path.exists(self.data_dir+'/'+self.dataset_name+'_part_assemblies_z.hdf5'):
            data_dict = h5py.File(self.data_dir+'/'+self.dataset_name+'_part_assemblies_z.hdf5', 'r')
            self.data_part_assemblies_z = data_dict['z'][:]
        else:
            print('Error: cannot load '+self.data_dir+'/'+self.dataset_name+'_part_assemblies_z.hdf5')
            exit(0)

        if os.path.exists(self.data_dir+'/'+self.dataset_name+'_suggestions_z.hdf5'):
            data_dict = h5py.File(self.data_dir+'/'+self.dataset_name+'_suggestions_z.hdf5', 'r')
            self.data_z_vectors = data_dict['z'][:]
        else:
            print('Error: cannot load '+self.data_dir+'/'+self.dataset_name+'_suggestions_z.hdf5')
            exit(0)

        self.build_model()

    @property
    def model_dir(self):
        return '{}_{}'.format(self.dataset_name, self.input_size)

    def build_model(self):
        self.part_assemblies_z = tf.placeholder(shape=[self.batch_size,self.z_dim], dtype=tf.float32)
        self.z_vector = tf.placeholder(shape=[self.batch_size,self.z_dim], dtype=tf.float32)
        self.noise = tf.placeholder(shape=[self.batch_size,self.noise_num,self.z_dim], dtype=tf.float32)

        self.G = self.generator(self.part_assemblies_z, self.noise, phase_train=True, reuse=False)
        self.sG = self.generator(self.part_assemblies_z, self.noise, phase_train=False, reuse=True)

        k = tf.math.argmin(tf.norm(tf.transpose(tf.transpose(self.G, [1, 0, 2])-self.z_vector, [1, 0, 2]), axis=2), axis=1)
        k = tf.reshape(k, [k.shape[0], 1])
        y = tf.gather_nd(self.G, k, batch_dims=1)
        self.loss = tf.reduce_mean(tf.square(self.z_vector - y))

        self.saver = tf.train.Saver(max_to_keep=10)

    def generator(self, z, noise, phase_train=True, reuse=False):
        with tf.variable_scope('generator') as scope:
            if reuse:
                scope.reuse_variables()

            z = tf.expand_dims(z, axis=1)
            zs = tf.tile(z, [1,self.noise_num,1])

            znoise = tf.concat([zs,noise],2)
            znoise = tf.reshape(znoise, [self.batch_size, znoise.shape[1]*znoise.shape[2]])
            print('znoise', znoise.shape)

            gen_lin1 = leaky_relu(fc(znoise, self.noise_num*self.f_dim, phase_train=phase_train, scope='gen_lin1'))
            gen_lin2 = leaky_relu(fc(gen_lin1, self.noise_num*self.f_dim, phase_train=phase_train, scope='gen_lin2'))
            gen_lin3 = leaky_relu(fc(gen_lin2, self.noise_num*self.f_dim, phase_train=phase_train, scope='gen_lin3'))
            gen_lin4 = leaky_relu(fc(gen_lin3, self.noise_num*self.f_dim, phase_train=phase_train, scope='gen_lin4'))
            gen_lin5 = leaky_relu(fc(gen_lin4, self.noise_num*self.f_dim, phase_train=phase_train, scope='gen_lin5'))
            ys = tf.nn.sigmoid(fc(gen_lin5, self.noise_num*self.z_dim, phase_train=phase_train, scope='gen_z'))
            ys = tf.reshape(ys, [self.batch_size, self.noise_num, self.z_dim])

            return ys

    def train(self, epoch_num, learning_rate, sample_dir='./sample/cIMLE-samples-train'):
        optim_cimle = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())

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

        batch_num = self.data_part_assemblies_z.shape[0] // self.batch_size

        avg_loss_cimle_list = []
        for epoch in range(counter, epoch_num):
            part_assemblies_z_train, z_vectors_train = shuffle(self.data_part_assemblies_z, self.data_z_vectors)
            avg_loss_cimle = 0
            for i in range(0, batch_num):
                part_assemblies_z_train_batch = part_assemblies_z_train[i*self.batch_size:(i+1)*self.batch_size]
                z_vectors_train_batch = z_vectors_train[i*self.batch_size:(i+1)*self.batch_size]
                noise_train_batch = np.random.normal(size=(self.batch_size, self.noise_num, self.z_dim))
                _, loss_cimle = sess.run([optim_cimle, self.loss],
                    feed_dict = {
                        self.part_assemblies_z: part_assemblies_z_train_batch,
                        self.z_vector: z_vectors_train_batch,
                        self.noise: noise_train_batch})
                avg_loss_cimle += loss_cimle
            avg_loss_cimle = avg_loss_cimle / batch_num
            print('Epoch: [%2d/%2d] time: %4.4f, cIMLE loss: %.8f' % (epoch, epoch_num, time.time() - start_time, avg_loss_cimle))

            self.save(self.checkpoint_dir, epoch)

            avg_loss_cimle_list.append(avg_loss_cimle)

        plt.figure()
        plt.plot(avg_loss_cimle_list)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig(os.path.join(sample_dir, 'cIMLE_loss.png'))
        plt.savefig(os.path.join(sample_dir, 'cIMLE_loss.pdf'))
        f = open(os.path.join(sample_dir, 'cIMLE_loss.txt'), 'w')
        for row in avg_loss_cimle_list:
            f.write(str(row) + '\n')
        f.close()

    def test(self, part_assemblies_z_test):
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            print(' [*] Load SUCCESS')
        else:
            print(' [!] Load failed...')
            return

        batch_num = part_assemblies_z_test.shape[0] // self.batch_size

        start_time = time.time()
        ys_list = []
        for i in range(0, batch_num):
            part_assemblies_z_test_batch = part_assemblies_z_test[i*self.batch_size:(i+1)*self.batch_size]
            noise_test_batch = np.random.normal(size=(self.batch_size, self.noise_num, self.z_dim))
            ys = self.sess.run(self.sG,
                feed_dict = {
                    self.part_assemblies_z: part_assemblies_z_test_batch,
                    self.noise: noise_test_batch})
            ys_list.append(ys)
        print('cIMLE', time.time() - start_time)

        return np.concatenate(ys_list)

    def save(self, checkpoint_dir, step):
        model_name = 'cIMLE.model'
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
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch_num', type=int, default=10000)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--noise_num', type=int, default=4)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    is_train = args.train
    batch_size = args.batch_size
    epoch_num = args.epoch_num
    learning_rate = args.learning_rate
    noise_num = args.noise_num

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        if is_train:
            cimle = cIMLE(sess=sess, batch_size=batch_size, noise_num=noise_num, dataset_name=dataset_name)
            cimle.train(epoch_num=epoch_num, learning_rate=learning_rate)
        else:
            print('Please use test_cIMLE.py for testing')
