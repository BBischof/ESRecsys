Spotify challenge
=================

In this directory we explore the Million Playlist Dataset from Spotify

Relevant links:

* [Spotify Blog](https://research.atspotify.com/2020/09/the-million-playlist-dataset-remastered/)

* [Dataset location](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge/dataset_files)

Setup
=====

Please download the million playlist data set and challenge into a directory called data

```
mkdir data
mkdir spotify_million_playlist_dataset
mkdir spotify_million_playlist_dataset_challenge

# Download both zip files using a browser into data


cd spotify_million_playlist_dataset_challenge
unzip ../spotify_million_playlist_dataset_challenge.zip

cd ../spotify_million_playlist_dataset
unzip ../spotify_million_dataset.zip

```

spotify_million_dataset.zip should be unpacked into data/spotify_million_playlist_dataset
spotify_million_dataset_challenge.zip should be unpacked into data/spotify_million_playlist_dataset_challenge

If you wish to install the GPU version of Jax please run

pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Making the dictionaries
=======================

python3 make_dictionary.py --playlists=data/spotify_million_playlist_dataset/data/mpd.slice*.json --output=data/dictionaries

A copy has been uploaded to weights and biases

wandb artifact put -n "recsys-spotify/dictionaries" -d "Dictionaries for tracks, arists and albums" -t "dictionaries" dictionaries

To fetch it run

wandb artifact get --type "dictionaries" recsys-spotify/dictionaries

Making the training data
========================

python3 make_training.py --playlists=data/spotify_million_playlist_dataset/data/mpd.slice.*.json --dictionaries=data/dictionaries

It should dump the following from the dictionaries that tell you how many of each URI is in the data:

2262292 tracks loaded
295860 artists loaded
734684 albums loaded

A copy has been uploaded to weights and biases

wandb artifact put -n "recsys-spotify/training" -d "Tensorflow example training" -t "tfrecord" training

To fetch it run
wandb artifact get --type "tfrecord" recsys-spotify/training

You might then have to move it to training or change the path
e.g.

 mv data/artifacts/training\:v1/* data/training