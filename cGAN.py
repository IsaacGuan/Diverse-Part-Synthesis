import os
import cv2
import h5py
import time
import mcubes
import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow_probability import distributions as tfd

from ops import *


class cGAN(object):
    def __init__(self, sess, batch_size, z_dim=128, f_dim=2048, checkpoint_dir='./checkpoint/cGAN', data_dir='./data', dataset_name='ChairPML'):
        self.sess = sess

        self.input_size = 64
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
        self.condition = tf.placeholder(shape=[self.batch_size,self.z_dim], dtype=tf.float32)
        self.z_vector = tf.placeholder(shape=[self.batch_size,self.z_dim], dtype=tf.float32)
        self.noise = tf.placeholder(shape=[self.batch_size,self.z_dim], dtype=tf.float32)

        self.G = self.generator(self.noise, self.condition, phase_train=True, reuse=False)
        self.D_real, self.D_logit_real = self.discriminator(self.z_vector, self.condition, phase_train=True, reuse=False)
        self.D_fake, self.D_logit_fake = self.discriminator(self.G, self.condition, phase_train=True, reuse=True)
        self.sG = self.generator(self.noise, self.condition, phase_train=False, reuse=True)

        self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logit_real, labels=tf.ones_like(self.D_logit_real)))
        self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logit_fake, labels=tf.zeros_like(self.D_logit_fake)))
        self.D_loss = self.D_loss_real + self.D_loss_fake
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logit_fake, labels=tf.ones_like(self.D_logit_fake)))

        self.vars = tf.trainable_variables()
        self.G_vars = [var for var in self.vars if 'gen_' in var.name or 'enc_' in var.name]
        self.D_vars = [var for var in self.vars if 'dis_' in var.name or 'enc_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=10)

    def generator(self, z, c, phase_train=True, reuse=False):
        with tf.variable_scope('generator') as scope:
            if reuse:
                scope.reuse_variables()

            zc = tf.concat(axis=-1, values=[z, c])

            gen_lin1 = leaky_relu(fc(zc, self.f_dim, phase_train=phase_train, scope='gen_lin1'))
            gen_lin2 = leaky_relu(fc(gen_lin1, self.f_dim, phase_train=phase_train, scope='gen_lin2'))

            z_hat = tf.nn.sigmoid(fc(gen_lin2, self.z_dim, phase_train=phase_train, scope='gen_z'))

            return z_hat
    
    def discriminator(self, z, c, phase_train=True, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            zc = tf.concat(axis=-1, values=[z, c])

            dis_lin1 = leaky_relu(fc(zc, self.f_dim, phase_train=phase_train, scope='dis_lin1'))
            dis_lin2 = leaky_relu(fc(dis_lin1, self.f_dim, phase_train=phase_train, scope='dis_lin2'))

            logit = fc(dis_lin2, 1, phase_train=phase_train, scope='dis_logit')
            prob = tf.nn.sigmoid(logit)
        
            return prob, logit

    def train(self, epoch_num, learning_rate, sample_dir='./sample/cGAN-samples-train'):
        D_optim = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5).minimize(self.D_loss, var_list=self.D_vars)
        G_optim = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5).minimize(self.G_loss, var_list=self.G_vars)
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

        avg_loss_D_list = []
        avg_loss_G_list = []
        for epoch in range(counter, epoch_num):
            avg_loss_D = 0
            avg_loss_G = 0
            for i in range(0, batch_num):
                part_assemblies_z_train_batch = self.data_part_assemblies_z[i*self.batch_size:(i+1)*self.batch_size]
                z_vectors_train_batch = self.data_z_vectors[i*self.batch_size:(i+1)*self.batch_size]
                noise_train_batch = np.random.normal(size=(self.batch_size, self.z_dim))
                _, loss_D = self.sess.run([D_optim, self.D_loss],
                    feed_dict = {
                        self.condition: part_assemblies_z_train_batch,
                        self.z_vector: z_vectors_train_batch,
                        self.noise: noise_train_batch})
                _, loss_G = self.sess.run([G_optim, self.G_loss],
                    feed_dict = {
                        self.condition: part_assemblies_z_train_batch,
                        self.noise: noise_train_batch})
                avg_loss_D += loss_D
                avg_loss_G += loss_G
            avg_loss_D = avg_loss_D / batch_num
            avg_loss_G = avg_loss_G / batch_num
            print('Epoch: [%2d/%2d] time: %4.4f, Discriminator loss: %.8f, Generator loss: %.8f' % (epoch, epoch_num, time.time() - start_time, avg_loss_D, avg_loss_G))

            self.save(self.checkpoint_dir, epoch)

            avg_loss_D_list.append(avg_loss_D)
            avg_loss_G_list.append(avg_loss_G)

        plt.figure()
        plt.plot(avg_loss_D_list)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig(os.path.join(sample_dir, 'Discriminator_loss.png'))
        plt.savefig(os.path.join(sample_dir, 'Discriminator_loss.pdf'))
        f = open(os.path.join(sample_dir, 'Discriminator_loss.txt'), 'w')
        for row in avg_loss_D_list:
            f.write(str(row) + '\n')
        f.close()

        plt.figure()
        plt.plot(avg_loss_G_list)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig(os.path.join(sample_dir, 'Generator_loss.png'))
        plt.savefig(os.path.join(sample_dir, 'Generator_loss.pdf'))
        f = open(os.path.join(sample_dir, 'Generator_loss.txt'), 'w')
        for row in avg_loss_G_list:
            f.write(str(row) + '\n')
        f.close()

    def test(self, part_assemblies_z_test, noise_num):
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            print(' [*] Load SUCCESS')
        else:
            print(' [!] Load failed...')
            return

        batch_num = part_assemblies_z_test.shape[0] // self.batch_size

        start_time = time.time()
        z_hat_list = []
        for i in range(0, batch_num):
            for j in range(0, noise_num):
                part_assemblies_z_test_batch = part_assemblies_z_test[i*self.batch_size:(i+1)*self.batch_size]
                noise_test_batch = np.random.normal(size=(self.batch_size, self.z_dim))
                z_hat = self.sess.run(self.sG,
                    feed_dict = {
                        self.condition: part_assemblies_z_test_batch,
                        self.noise: noise_test_batch})
                z_hat_list.append(z_hat)
        print('cGAN', time.time() - start_time)

        return np.concatenate(z_hat_list)

    def save(self, checkpoint_dir, step):
        model_name = 'cGAN.model'
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
    parser.add_argument('--epoch_num', type=int, default=2000)
    parser.add_argument('--learning_rate', type=float, default=0.00001)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    is_train = args.train
    epoch_num = args.epoch_num
    learning_rate = args.learning_rate

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        cgan = cGAN(sess=sess, batch_size=64, dataset_name=dataset_name)
        if is_train:
            cgan.train(epoch_num=epoch_num, learning_rate=learning_rate)
        else:
            z_hat = cgan.test(part_assemblies_z_test=cgan.data_part_assemblies_z, noise_num=1)
