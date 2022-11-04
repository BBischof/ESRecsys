Installation instructions
=========================

This tutorial has been tested on Ubuntu (under Windows, WSL2 Ubuntu).

OS level installs
=================
Some packages require python 3.9 so please ensure that you are not using python 3.10

To get python 3.9 and the virtual environment on Ubuntu

sudo apt-install python3.9
sudo apt-install python3.9-venv

Python virtual environment setup
================================

To setup the python virutal environment, in the parent directory above this one run:

python3 -m venv venv_name

cd venv_name

source bin/activate

Then go to a code directory e.g.

cd pinterest

pip install -r requirements.txt

If you want the GPU version of jax please follow the latest instructions on the jax webpage.

If you wish to skip all the data processing steps and skip straight to training embeddings you can make use of the pre-made weights and biases artifacts.
Skip to the train word embeddings section.

Getting the Data
================

For this data set we will be using the Shop The Look Data set from

[Shop the Look data set](https://github.com/kang205/STL-Dataset)

[Wang-Cheng Kang](http://kwc-oliver.com), Eric Kim, [Jure Leskovec](https://cs.stanford.edu/people/jure/), Charles Rosenberg, [Julian McAuley](http://cseweb.ucsd.edu/~jmcauley/) (2019). *[Complete the Look: Scene-based Complementary Product Recommendation.](https://arxiv.org/pdf/1812.01748.pdf)* In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR'19)

A copy of the STL dataset is in the directory STL-Dataset.
The originals can be fetched directly a the github repo above.

Running sample code
===================

To run a random item recommender:

python3 random_item_recommender.py --input_file=STL-Dataset/fashion-cat.json --output_html=output.html

To view the output open output.html with your favorite browser

e.g. google-chrome output.html

To fetch the images from pinterest for training, this will take a while!
Pinterest might also block you for scraping if you don't set the sleep timeout to a high enough value.

python3 fetch_images.py --input_file=STL-Dataset/fashion.json --output_dir=images/ --max_lines=100000

Alternatively if you have a weights and biases account:

 wandb artifact get building-recsys/recsys-pinterest/shop_the_look:v1

The images will then be in

./artifacts/shop_the_look:v1

Training the Model
==================

python3 train_shop_the_look.py --input_file=STL-Dataset/fashion.json --image_dir=./artifacts/shop_the_look\:v1