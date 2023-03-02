# FluvialGAN3D
This is a supporting repositories of our manuscript 'A Conditional GAN-based Approach to Build 3D Facies Models Sequentially Upwards' submitted to Computers & Geosciences

The dataset is available at https://github.com/GeoDataScienceUQ/GANRiverI

The Fluvial GAN for 2D simulation is available at https://github.com/GeoDataScienceUQ/Fluvial_GAN

To train your own version, please download the dataset (this work uses the 7-facies version as default) and set up your path.

Then run 'train_3d.py' in the fold that you'd like to try.

After training both 2D and 3D models, use the simulator in 'test_all.py' to start random simulation or feed your latent/noise vectors

## Acknowledgement
This code borrows heavily from https://github.com/NVlabs/SPADE
