# Diverse Part Synthesis

This repository contains the experimental code for synthesizing shapes via incremental part suggestion, as described in the paper ["Diverse Part Synthesis for 3D Shape Creation"](https://arxiv.org/abs/2401.09384).

## Data

The experimental datasets are available [here](https://drive.google.com/drive/folders/1fGr3b3AvljxpralMToh6s_DHZVP16Ji4?usp=sharing). Please download and unzip each of them directly into the `data` folder.

## Dependencies

All the dependencies are packaged in `environment.yml` that can be installed using [conda](https://docs.conda.io/).
```bash
conda env create -f environment.yml
```
The conda environment can then be activated.
```bash
conda activate dps
```
All the following commands are run in this conda environment.

## Training

To train the PCN and get the latent space of shape parts, please run the following commands.
```bash
python PCN.py --train
python get_z.py --train
```
To train the implicit decoder, please run the following commands.
```bash
python IMDecoder.py --train --epoch 250 --real_size 16 --batch_size_input 4096
python IMDecoder.py --train --epoch 500 --real_size 32 --batch_size_input 8192
python IMDecoder.py --train --epoch 1000 --real_size 64 --batch_size_input 32768
```
Prior to training the PSN, training data for the PSN can be generated using the following commands.
```bash
python generate_assembly_suggestion_pairs.py
python get_z_for_suggestions.py
python get_z_for_part_assemblies.py
```
To train all the implementations of PSN, please run the following commands.
```bash
python MDN.py --train
python cGAN.py --train
python cIMLE.py --train
python cDDPM.py --train
```

## Shape Synthesis

To iteratively synthesize shapes using cIMLE and cDDPM, please run the following commands.
```bash
python iterative_generation_cIMLE.py
python iterative_generation_cDDPM.py
```
