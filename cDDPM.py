import os
import h5py
import torch
import random
import argparse
import numpy as np

from conditional_denoising_diffusion_pytorch_1d import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D

DATA_DIR = './data'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='ChairPML')
    parser.add_argument('--train', action='store_true', help='train or test')
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=8)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    is_train = args.train
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size

    checkpoint_dir = './checkpoint/cDDPM/'+dataset_name

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

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

    if is_train:
        loss = diffusion(training_seq[:,0:1,:],training_seq[:,1:2,:])
        loss.backward()

    trainer = Trainer1D(
        diffusion,
        dataset = dataset,
        train_batch_size = train_batch_size,
        train_lr = 8e-5,
        train_num_steps = 500000,
        gradient_accumulate_every = 2,
        ema_decay = 0.995,
        amp = True,
        results_folder = checkpoint_dir
    )

    if is_train:
        trainer.train()
    else:
        sample_dir = './sample/cDDPM-samples-test/'+dataset_name
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

        trainer.load('500')
        idx = random.randrange(part_assemblies_z.shape[0])
        condition = torch.repeat_interleave(part_assemblies_z_training_seq[idx:idx+1], test_batch_size, dim=0).cuda()
        sampled_seq = diffusion.sample(condition, batch_size=test_batch_size)
        sampled_z = sampled_seq.cpu().detach().numpy().squeeze()
        np.savetxt(sample_dir+'/'+str(idx)+'_sampled_z.txt', sampled_z)
