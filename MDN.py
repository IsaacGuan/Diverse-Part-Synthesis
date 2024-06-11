import os
import h5py
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, BatchNormalization, Conv3D, Dense, Reshape, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import backend as K
from tensorflow_probability import distributions as tfd


class MDN(object):
    def __init__(self, voxel_size=64, z_dim=128, cat_num=4, checkpoint_dir='./checkpoint/MDN', data_dir='./data', dataset_name='ChairPML'):
        self.z_dim = z_dim
        self.cat_num = cat_num

        self.input_shape = (1, z_dim)
        self.z_shape = (1, z_dim)

        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir
        self.dataset_name = dataset_name

        if os.path.exists(self.data_dir+'/'+self.dataset_name+'_part_assemblies_z.hdf5'):
            data_dict = h5py.File(self.data_dir+'/'+self.dataset_name+'_part_assemblies_z.hdf5', 'r')
            self.data_part_assemblies_z = data_dict['z'][:]
            self.data_part_assemblies_z = np.expand_dims(self.data_part_assemblies_z, axis=1)
        else:
            print('Error: cannot load '+self.data_dir+'/'+self.dataset_name+'_part_assemblies_z.hdf5')
            exit(0)

        if os.path.exists(self.data_dir+'/'+self.dataset_name+'_suggestions_z.hdf5'):
            data_dict = h5py.File(self.data_dir+'/'+self.dataset_name+'_suggestions_z.hdf5', 'r')
            self.data_z_vectors = data_dict['z'][:]
            self.data_z_vectors = np.expand_dims(self.data_z_vectors, axis=1)
        else:
            print('Error: cannot load '+self.data_dir+'/'+self.dataset_name+'_suggestions_z.hdf5')
            exit(0)

    def elu_modified(self, x):
        return tf.nn.elu(x) + 1 + K.epsilon()

    def get_model(self):
        l_in = Input(shape=self.input_shape)
        l_z = Input(shape=self.z_shape)

        l_fc1 = Dense(
            units = 1024,
            activation = 'tanh')(l_in)
        l_fc2 = Dense(
            units = 1024,
            activation = 'tanh')(l_fc1)
        l_fc3 = Dense(
            units = 1024,
            activation = 'tanh')(l_fc2)

        l_alpha = Dense(
            units = self.cat_num,
            activation = 'softmax')(l_fc3)
        l_mu = Dense(
            units = self.cat_num*self.z_dim)(l_fc3)
        l_sigma = Dense(
            units = self.cat_num*self.z_dim,
            activation = self.elu_modified)(l_fc3)

        return {'inputs': l_in,
                'z': l_z,
                'alpha': l_alpha,
                'mu': l_mu,
                'sigma': l_sigma}

    def mdn_loss(self, z, alpha, mu, sigma):
        alpha = K.repeat_elements(alpha, self.z_dim, axis=1)
        alpha = K.expand_dims(alpha, axis=3)

        mu = K.reshape(mu, (tf.shape(mu)[0], self.z_dim, self.cat_num))
        mu = K.expand_dims(mu, axis=3)

        sigma = K.reshape(sigma, (tf.shape(sigma)[0], self.z_dim, self.cat_num))
        sigma = K.expand_dims(sigma, axis=3)

        gm = tfd.MixtureSameFamily(
            mixture_distribution = tfd.Categorical(probs=alpha),
            components_distribution = tfd.Normal(
                loc = mu,
                scale = sigma))

        z = tf.transpose(z, (0, 2, 1))

        return tf.reduce_mean(-gm.log_prob(z))

    def train(self, learning_rate, batch_size, epoch_num, sample_dir='./sample/MDN-samples-train'):
        model = self.get_model()

        inputs = model['inputs']
        z = model['z']
        alpha = model['alpha']
        mu = model['mu']
        sigma = model['sigma']

        model_train = Model([inputs, z], [alpha, mu, sigma])

        model_train.add_loss(self.mdn_loss(z, alpha, mu, sigma))

        adam = Adam(lr=learning_rate)
        model_train.compile(optimizer=adam)

        sample_dir = os.path.join(sample_dir, self.dataset_name)
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

        part_assemblies_z_train = self.data_part_assemblies_z
        z_train = self.data_z_vectors

        results = model_train.fit(
            [part_assemblies_z_train, z_train],
            batch_size = batch_size,
            epochs = epoch_num,
            validation_data = ([part_assemblies_z_train, z_train], None)
        )

        plt.figure()
        plt.plot(results.history['loss'])
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig(os.path.join(sample_dir, 'MDN_loss.png'))
        plt.savefig(os.path.join(sample_dir, 'MDN_loss.pdf'))
        f = open(os.path.join(sample_dir, 'MDN_loss.txt'), 'w')
        for row in results.history['loss']:
            f.write(str(row) + '\n')
        f.close()

        checkpoint_dir = os.path.join(self.checkpoint_dir, self.dataset_name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        model_train.save_weights(os.path.join(checkpoint_dir, 'MDN.hdf5'))

    def test(self, part_assemblies_z_test):
        model = self.get_model()

        model_test = Model(model['inputs'], [model['alpha'], model['mu'], model['sigma']])
        model_test.load_weights(os.path.join(self.checkpoint_dir, self.dataset_name, 'MDN.hdf5'))

        start_time = time.time()
        predictions = model_test.predict(part_assemblies_z_test)
        print('MDN', time.time() - start_time)

        predictions_alpha = predictions[0]
        predictions_mu = predictions[1]
        predictions_sigma = predictions[2]

        predictions_mu = np.reshape(predictions_mu, (np.shape(predictions_mu)[0], self.z_dim, self.cat_num))
        predictions_sigma = np.reshape(predictions_sigma, (np.shape(predictions_mu)[0], self.z_dim, self.cat_num))

        return predictions_alpha, predictions_mu, predictions_sigma


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='ChairPML')
    parser.add_argument('--train', action='store_true', help='train or test')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epoch_num', type=int, default=1000)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    is_train = args.train
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epoch_num = args.epoch_num

    mdn = MDN(dataset_name=dataset_name)

    if is_train:
        mdn.train(learning_rate=learning_rate, batch_size=batch_size, epoch_num=epoch_num)
    else:
        mdn.test(part_assemblies_z_test=mdn.data_part_assemblies_z)
