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
from transformer import *
from sklearn.utils import shuffle


class PCN(object):
    def __init__(self, sess, batch_size, is_training=False, z_dim=128, f_dim=32, checkpoint_dir='./checkpoint/PCN', data_dir='./data', dataset_name='ChairPML'):
        self.sess = sess

        self.input_size = 64
        self.batch_size = batch_size

        self.z_dim = z_dim
        self.f_dim = f_dim

        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir
        self.dataset_name = dataset_name

        if is_training:
            phase = 'train'
        else:
            phase = 'test'

        self.n_parts_list = []
        data_parts = []
        data_parts_t = []
        data_scales = []
        data_translations = []
        parts_info = load_part_data_info(dataset_name, phase)
        curr_shape_path = ''
        for part_info in parts_info:
            shape_path, part_idx = part_info
            n_parts, part_voxel, part_voxel_t, scale, translation = load_from_hdf5_for_PCN(shape_path, part_idx)
            if shape_path != curr_shape_path:
                self.n_parts_list.append(n_parts)
            curr_shape_path = shape_path
            data_parts.append(part_voxel)
            data_parts_t.append(part_voxel_t)
            data_scales.append(scale)
            data_translations.append(translation)
        self.data_parts = np.expand_dims(np.array(data_parts), axis=-1)
        self.data_parts_t = np.expand_dims(np.array(data_parts_t), axis=-1)
        self.data_scales = np.array(data_scales).squeeze()
        self.data_translations = np.array(data_translations)

        self.build_model()

    @property
    def model_dir(self):
        return '{}_{}'.format(self.dataset_name, self.input_size)

    def build_model(self):
        self.parts = tf.placeholder(shape=[self.batch_size,self.input_size,self.input_size,self.input_size,1], dtype=tf.float32)
        self.z_vector = tf.placeholder(shape=[self.batch_size,self.z_dim], dtype=tf.float32)

        self.E = self.encoder(self.parts, phase_train=True, reuse=False)
        self.D = self.decoder(self.E, phase_train=True, reuse=False)
        self.sE = self.encoder(self.parts, phase_train=False, reuse=True)
        self.sD = self.decoder(self.sE, phase_train=False, reuse=True)
        self.ZD = self.decoder(self.z_vector, phase_train=False, reuse=True)

        self.loss_ae = tf.reduce_mean(tf.square(self.parts - self.D))

        self.parts_t = tf.placeholder(shape=[self.batch_size,self.input_size,self.input_size,self.input_size,1], dtype=tf.float32)
        self.scale = tf.placeholder(shape=[self.batch_size], dtype=tf.float32)
        self.trans = tf.placeholder(shape=[self.batch_size,3], dtype=tf.float32)

        self.w_scale, self.w_trans = self.localize(self.sD, phase_train=True, reuse=False)
        self.w_parts_t = transform(self.sD, self.w_scale, self.w_trans)
        self.w_scale_s, self.w_trans_s = self.localize(self.sD, phase_train=False, reuse=True)
        self.w_parts_t_s = transform(self.parts, self.w_scale_s, self.w_trans_s)
        self.w_scale_z, self.w_trans_z = self.localize(self.ZD, phase_train=False, reuse=True)
        self.w_parts_t_z = transform(self.ZD, self.w_scale_z, self.w_trans_z)

        self.loss_rec = tf.reduce_mean(tf.square(self.parts_t - self.w_parts_t))
        self.loss_st = self.dis(self.scale, self.w_scale) + self.dis(self.trans, self.w_trans)
        self.loss_stn = self.loss_rec + self.loss_st

        self.saver = tf.train.Saver(max_to_keep=10)

    def dis(self, o1, o2):
        eucd2 = tf.pow(tf.subtract(o1, o2), 2)
        eucd2 = tf.reduce_sum(eucd2)
        eucd = tf.sqrt(eucd2+1e-6, name='eucd')
        loss = tf.reduce_mean(eucd, name='loss')
        return loss

    def encoder(self, x, phase_train=True, reuse=False):
        with tf.variable_scope('encoder') as scope:
            if reuse:
                scope.reuse_variables()

            enc_conv1 = conv3d(x, self.f_dim, phase_train=phase_train, scope='enc_conv1')
            enc_conv1 = leaky_relu(batch_norm(enc_conv1, phase_train))

            enc_conv2 = conv3d(enc_conv1, self.f_dim*2, phase_train=phase_train, scope='enc_conv2')
            enc_conv2 = leaky_relu(batch_norm(enc_conv2, phase_train))

            enc_conv3 = conv3d(enc_conv2, self.f_dim*4, phase_train=phase_train, scope='enc_conv3')
            enc_conv3 = leaky_relu(batch_norm(enc_conv3, phase_train))

            enc_conv4 = conv3d(enc_conv3, self.f_dim*8, phase_train=phase_train, scope='enc_conv4')
            enc_conv4 = leaky_relu(batch_norm(enc_conv4, phase_train))

            enc_conv5 = conv3d(enc_conv4, 128, phase_train=phase_train, scope='enc_conv5', d_h=1, d_w=1, d_d=1)
            enc_conv5 = tf.reshape(enc_conv5, [self.batch_size,128*4*4*4])

            z = tf.nn.sigmoid(fc(enc_conv5, self.z_dim, phase_train=phase_train, scope='enc_z'))

            return z

    def decoder(self, z, phase_train=True, reuse=False):
        with tf.variable_scope('decoder') as scope:
            if reuse:
                scope.reuse_variables()

            dec_lin1 = leaky_relu(fc(z, 128*4*4*4, phase_train=phase_train, scope='dec_lin1'))
            dec_lin1 = tf.reshape(dec_lin1, [self.batch_size,4,4,4,128])

            dec_deconv1 = deconv3d(dec_lin1, self.f_dim*8, output_shape=[self.batch_size,4,4,4,256], phase_train=phase_train, scope='dec_deconv1', d_h=1, d_w=1, d_d=1)
            dec_deconv1 = leaky_relu(batch_norm(dec_deconv1, phase_train))

            dec_deconv2 = deconv3d(dec_deconv1, self.f_dim*4, output_shape=[self.batch_size,8,8,8,128], phase_train=phase_train, scope='dec_deconv2')
            dec_deconv2 = leaky_relu(batch_norm(dec_deconv2, phase_train))

            dec_deconv3 = deconv3d(dec_deconv2, self.f_dim*2, output_shape=[self.batch_size,16,16,16,64], phase_train=phase_train, scope='dec_deconv3')
            dec_deconv3 = leaky_relu(batch_norm(dec_deconv3, phase_train))

            dec_deconv4 = deconv3d(dec_deconv3, self.f_dim, output_shape=[self.batch_size,32,32,32,32], phase_train=phase_train, scope='dec_deconv4')
            dec_deconv4 = leaky_relu(batch_norm(dec_deconv4, phase_train))

            dec_deconv5 = deconv3d(dec_deconv4, 1, output_shape=[self.batch_size,64,64,64,1], phase_train=phase_train, scope='dec_deconv5')

            recon = batch_norm(dec_deconv5, phase_train)

            return recon

    def localize(self, x, phase_train=True, reuse=False):
        with tf.variable_scope('localize') as scope:
            if reuse:
                scope.reuse_variables()

            loc_conv1 = conv3d(x, self.f_dim, phase_train, 'loc_conv1')
            loc_conv1 = tf.nn.relu(loc_conv1)

            loc_conv2 = conv3d(loc_conv1, self.f_dim*2, phase_train, 'loc_conv2')
            loc_conv2 = tf.nn.relu(batch_norm(loc_conv2, phase_train))

            loc_conv3 = conv3d(loc_conv2, self.f_dim*4, phase_train, 'loc_conv3')
            loc_conv3 = tf.nn.relu(batch_norm(loc_conv3, phase_train))

            loc_conv4 = conv3d(loc_conv3, self.f_dim*8, phase_train, 'loc_conv4')
            loc_conv4 = tf.nn.relu(batch_norm(loc_conv4, phase_train))

            loc_fc1 = tf.reshape(loc_conv4, [self.batch_size, -1])
            loc_fc1 = fc(loc_fc1, 128, phase_train, 'loc_fc1')
            loc_fc1 = tf.nn.relu(batch_norm(loc_fc1, phase_train))

            loc_fc2 = fc(loc_fc1, self.f_dim*2, phase_train, 'loc_fc2')
            loc_fc2 = tf.nn.relu(loc_fc2)

            loc_fc3 = fc(loc_fc2, self.f_dim*2, phase_train, 'loc_fc3')
            loc_fc3 = tf.nn.relu(loc_fc3)

            loc_fc4 = fc(loc_fc3, 4, phase_train, 'loc_fc4')

            scale = loc_fc4[:,0]
            translation = loc_fc4[:,1:]

        return scale, translation

    def train(self, epoch_num, learning_rate_ae, learning_rate_stn, sample_dir='./sample/PCN-samples-train'):
        optim_ae = tf.train.AdamOptimizer(learning_rate=learning_rate_ae, beta1=0.5).minimize(self.loss_ae)
        optim_stn = tf.train.AdamOptimizer(learning_rate=learning_rate_stn, beta1=0.5).minimize(self.loss_stn)
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

        batch_num = self.data_parts.shape[0] // self.batch_size

        avg_loss_ae_list = []
        avg_loss_rec_list = []
        avg_loss_st_list = []
        for epoch in range(counter, epoch_num):
            parts_train, parts_t_train, scales_train, translations_train = shuffle(self.data_parts, self.data_parts_t, self.data_scales, self.data_translations)
            avg_loss_ae = 0
            avg_loss_rec = 0
            avg_loss_st = 0
            for i in range(0, batch_num):
                parts_train_batch = parts_train[i*self.batch_size:(i+1)*self.batch_size]
                parts_t_train_batch = parts_t_train[i*self.batch_size:(i+1)*self.batch_size]
                scales_train_batch = scales_train[i*self.batch_size:(i+1)*self.batch_size]
                translations_train_batch = translations_train[i*self.batch_size:(i+1)*self.batch_size]

                _, loss_ae = sess.run([optim_ae, self.loss_ae], feed_dict={self.parts: parts_train_batch})
                _, loss_rec, loss_st, loss_stn = self.sess.run([optim_stn, self.loss_rec, self.loss_st, self.loss_stn],
                    feed_dict={
                        self.parts: parts_train_batch,
                        self.parts_t: parts_t_train_batch,
                        self.scale: scales_train_batch,
                        self.trans: translations_train_batch
                    }
                )
                avg_loss_ae += loss_ae
                avg_loss_rec += loss_rec
                avg_loss_st += loss_st
            avg_loss_ae = avg_loss_ae / batch_num
            avg_loss_rec = avg_loss_rec / batch_num
            avg_loss_st = avg_loss_st / batch_num
            print('Epoch: [%2d/%2d] time: %4.4f, AE loss: %.8f, reconstruction loss: %.8f, scale and translation loss: %.8f' % (epoch, epoch_num, time.time() - start_time, avg_loss_ae, avg_loss_rec, avg_loss_st))

            batch_idx = random.randrange(batch_num)
            idx = random.randrange(self.batch_size)
            samples = self.sess.run(self.w_parts_t_s, feed_dict={self.parts:parts_train[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]})
            samples = samples[...,0]
            sample = samples[idx]
            real_samples = parts_t_train[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
            real_samples = real_samples[...,0]
            real_sample = real_samples[idx]

            img1 = np.clip(np.amax(sample, axis=0)*256, 0, 255).astype(np.uint8)
            img2 = np.clip(np.amax(sample, axis=1)*256, 0, 255).astype(np.uint8)
            img3 = np.clip(np.amax(sample, axis=2)*256, 0, 255).astype(np.uint8)
            cv2.imwrite(sample_dir+'/'+str(epoch)+'_1.png', img1)
            cv2.imwrite(sample_dir+'/'+str(epoch)+'_2.png', img2)
            cv2.imwrite(sample_dir+'/'+str(epoch)+'_3.png', img3)
            img1 = np.clip(np.amax(real_sample, axis=0)*256, 0, 255).astype(np.uint8)
            img2 = np.clip(np.amax(real_sample, axis=1)*256, 0, 255).astype(np.uint8)
            img3 = np.clip(np.amax(real_sample, axis=2)*256, 0, 255).astype(np.uint8)
            cv2.imwrite(sample_dir+'/'+str(epoch)+'_1_gt.png', img1)
            cv2.imwrite(sample_dir+'/'+str(epoch)+'_2_gt.png', img2)
            cv2.imwrite(sample_dir+'/'+str(epoch)+'_3_gt.png', img3)

            self.save(self.checkpoint_dir, epoch)

            avg_loss_ae_list.append(avg_loss_ae)
            avg_loss_rec_list.append(avg_loss_rec)
            avg_loss_st_list.append(avg_loss_st)

        plt.figure()
        plt.plot(avg_loss_ae_list)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig(os.path.join(sample_dir, 'AE_loss.png'))
        plt.savefig(os.path.join(sample_dir, 'AE_loss.pdf'))
        f = open(os.path.join(sample_dir, 'AE_loss.txt'), 'w')
        for row in avg_loss_ae_list:
            f.write(str(row) + '\n')
        f.close()

        plt.figure()
        plt.plot(avg_loss_rec_list)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig(os.path.join(sample_dir, 'rec_loss.png'))
        plt.savefig(os.path.join(sample_dir, 'rec_loss.pdf'))
        f = open(os.path.join(sample_dir, 'rec_loss.txt'), 'w')
        for row in avg_loss_rec_list:
            f.write(str(row) + '\n')
        f.close()

        plt.figure()
        plt.plot(avg_loss_st_list)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig(os.path.join(sample_dir, 'st_loss.png'))
        plt.savefig(os.path.join(sample_dir, 'st_loss.pdf'))
        f = open(os.path.join(sample_dir, 'st_loss.txt'), 'w')
        for row in avg_loss_st_list:
            f.write(str(row) + '\n')
        f.close()

    def train_st(self, epoch_num, learning_rate_st, sample_dir='./sample/PCN-samples-train'):
        optim_st = tf.train.AdamOptimizer(learning_rate=learning_rate_st, beta1=0.5).minimize(self.loss_st)
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

        batch_num = self.data_parts.shape[0] // self.batch_size

        avg_loss_st_list = []
        for epoch in range(counter, epoch_num):
            parts_train, parts_t_train, scales_train, translations_train = shuffle(self.data_parts, self.data_parts_t, self.data_scales, self.data_translations)
            avg_loss_st = 0
            for i in range(0, batch_num):
                parts_train_batch = parts_train[i*self.batch_size:(i+1)*self.batch_size]
                parts_t_train_batch = parts_t_train[i*self.batch_size:(i+1)*self.batch_size]
                scales_train_batch = scales_train[i*self.batch_size:(i+1)*self.batch_size]
                translations_train_batch = translations_train[i*self.batch_size:(i+1)*self.batch_size]

                _, loss_st = self.sess.run([optim_st, self.loss_st],
                    feed_dict={
                        self.parts: parts_train_batch,
                        self.scale: scales_train_batch,
                        self.trans: translations_train_batch
                    }
                )
                avg_loss_st += loss_st
            avg_loss_st = avg_loss_st / batch_num
            print('Epoch: [%2d/%2d] time: %4.4f,scale and translation loss: %.8f' % (epoch, epoch_num, time.time() - start_time, avg_loss_st))

            batch_idx = random.randrange(batch_num)
            idx = random.randrange(self.batch_size)
            samples = self.sess.run(self.w_parts_t_s, feed_dict={self.parts:parts_train[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]})
            samples = samples[...,0]
            sample = samples[idx]
            real_samples = parts_t_train[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
            real_samples = real_samples[...,0]
            real_sample = real_samples[idx]

            img1 = np.clip(np.amax(sample, axis=0)*256, 0, 255).astype(np.uint8)
            img2 = np.clip(np.amax(sample, axis=1)*256, 0, 255).astype(np.uint8)
            img3 = np.clip(np.amax(sample, axis=2)*256, 0, 255).astype(np.uint8)
            cv2.imwrite(sample_dir+'/'+str(epoch)+'_1.png', img1)
            cv2.imwrite(sample_dir+'/'+str(epoch)+'_2.png', img2)
            cv2.imwrite(sample_dir+'/'+str(epoch)+'_3.png', img3)
            img1 = np.clip(np.amax(real_sample, axis=0)*256, 0, 255).astype(np.uint8)
            img2 = np.clip(np.amax(real_sample, axis=1)*256, 0, 255).astype(np.uint8)
            img3 = np.clip(np.amax(real_sample, axis=2)*256, 0, 255).astype(np.uint8)
            cv2.imwrite(sample_dir+'/'+str(epoch)+'_1_gt.png', img1)
            cv2.imwrite(sample_dir+'/'+str(epoch)+'_2_gt.png', img2)
            cv2.imwrite(sample_dir+'/'+str(epoch)+'_3_gt.png', img3)

            self.save(self.checkpoint_dir, epoch)

            avg_loss_st_list.append(avg_loss_st)

        plt.figure()
        plt.plot(avg_loss_st_list)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig(os.path.join(sample_dir, 'st_loss_retrain.png'))
        plt.savefig(os.path.join(sample_dir, 'st_loss_retrain.pdf'))
        f = open(os.path.join(sample_dir, 'st_loss_retrain.txt'), 'w')
        for row in avg_loss_st_list:
            f.write(str(row) + '\n')
        f.close()

    def test(self, sample_dir='./sample/PCN-samples-test'):
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            print(' [*] Load SUCCESS')
        else:
            print(' [!] Load failed...')
            return

        sample_dir = os.path.join(sample_dir, self.model_dir)
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

        curr_part_num = 0
        for i in range(len(self.n_parts_list)):
            sample_list = []
            gt_list = []
            for j in range(self.n_parts_list[i]):
                scale, translation, sample = self.sess.run([self.w_scale_s, self.w_trans_s, self.w_parts_t_s],
                    feed_dict = {
                        self.parts: self.data_parts[curr_part_num+j:curr_part_num+j+1]
                    })
                sample_list.append(sample)
                gt_list.append(self.data_parts_t[curr_part_num+j:curr_part_num+j+1])
            curr_part_num += self.n_parts_list[i]
            if not os.path.exists(sample_dir+'/'+str(i)):
                os.makedirs(sample_dir+'/'+str(i))
            for j in range(len(sample_list)):
                thres = 0.5
                vertices, triangles = mcubes.marching_cubes(sample_list[j].squeeze(), thres)
                mcubes.export_mesh(vertices, triangles, sample_dir+'/'+str(i)+'/'+str(j)+'.dae')
                vertices, triangles = mcubes.marching_cubes(gt_list[j].squeeze(), thres)
                mcubes.export_mesh(vertices, triangles, sample_dir+'/'+str(i)+'/'+str(j)+'_gt.dae')

                np.save(sample_dir+'/'+str(i)+'/'+str(j)+'.npy', sample_list[j].squeeze())

                img1 = np.clip(np.amax(sample_list[j].squeeze(), axis=0)*256, 0,255).astype(np.uint8)
                img2 = np.clip(np.amax(sample_list[j].squeeze(), axis=1)*256, 0,255).astype(np.uint8)
                img3 = np.clip(np.amax(sample_list[j].squeeze(), axis=2)*256, 0,255).astype(np.uint8)
                cv2.imwrite(sample_dir+'/'+str(i)+'/'+str(j)+'_1.png',img1)
                cv2.imwrite(sample_dir+'/'+str(i)+'/'+str(j)+'_2.png',img2)
                cv2.imwrite(sample_dir+'/'+str(i)+'/'+str(j)+'_3.png',img3)

                print('[sample]')

    def test_parts(self, batch_parts):
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            print(' [*] Load SUCCESS')
        else:
            print(' [!] Load failed...')
            return

        batch_num = batch_parts.shape[0] // self.batch_size

        samples_list = []
        scales_list = []
        translations_list = []
        samples_T_list = []
        for i in range(0, batch_num):
            samples, scales, translations, samples_T = self.sess.run([self.sD, self.w_scale_s, self.w_trans_s, self.w_parts_t_s],
                feed_dict={
                    self.parts: batch_parts[i*self.batch_size:(i+1)*self.batch_size]
                })
            samples_list.append(samples)
            scales_list.append(scales)
            translations_list.append(translations)
            samples_T_list.append(samples_T)

        return np.concatenate(samples_list), np.concatenate(scales_list), np.concatenate(translations_list), np.concatenate(samples_T_list)

    def test_z(self, batch_z):
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            print(' [*] Load SUCCESS')
        else:
            print(' [!] Load failed...')
            return

        batch_num = batch_z.shape[0]//self.batch_size

        samples_list = []
        scales_list = []
        translations_list = []
        samples_T_list = []
        for i in range(0, batch_num):
            samples, scales, translations, samples_T = self.sess.run([self.ZD, self.w_scale_z, self.w_trans_z, self.w_parts_t_z],
                feed_dict={
                    self.z_vector: batch_z[i*self.batch_size:(i+1)*self.batch_size]
                })
            samples_list.append(samples)
            scales_list.append(scales)
            translations_list.append(translations)
            samples_T_list.append(samples_T)

        return np.concatenate(samples_list), np.concatenate(scales_list), np.concatenate(translations_list), np.concatenate(samples_T_list)

    def get_z(self, batch_parts, dataset_name, z_dir='./data'):
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            print(' [*] Load SUCCESS')
        else:
            print(' [!] Load failed...')
            return

        hdf5_path = z_dir+'/'+dataset_name+'_z.hdf5'
        hdf5_file = h5py.File(hdf5_path, mode='w')
        hdf5_file.create_dataset('z', [batch_parts.shape[0],self.z_dim], np.float32)

        batch_num = batch_parts.shape[0] // self.batch_size

        for i in range(0, batch_num):
            z = self.sess.run(self.sE,
                feed_dict={
                    self.parts: batch_parts[i*self.batch_size:(i+1)*self.batch_size],
                })
            hdf5_file['z'][i*self.batch_size:(i+1)*self.batch_size,:] = z
            print('[z]')

        hdf5_file.close()

    def save(self, checkpoint_dir, step):
        model_name = 'PCN.model'
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
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch_num', type=int, default=20000)
    parser.add_argument('--learning_rate_ae', type=float, default=0.0001)
    parser.add_argument('--learning_rate_stn', type=float, default=0.000001)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    is_training = args.train
    batch_size = args.batch_size
    epoch_num = args.epoch_num
    learning_rate_ae = args.learning_rate_ae
    learning_rate_stn = args.learning_rate_stn

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        if is_training:
            pcn = PCN(sess=sess, batch_size=batch_size, is_training=is_training, dataset_name=dataset_name)
            pcn.train(epoch_num=epoch_num, learning_rate_ae=learning_rate_ae, learning_rate_stn=learning_rate_stn)
            pcn.train_st(epoch_num=epoch_num+1000, learning_rate_st=learning_rate_stn*0.1)
        else:
            pcn = PCN(sess=sess, batch_size=1, is_training=is_training, dataset_name=dataset_name)
            pcn.test()
            # pcn.get_z(batch_parts=pcn.data_parts, dataset_name=dataset_name)
