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


Making the dictionaries
=======================

python3 make_dictionary.py --playlists=data/spotify_million_playlist_dataset/data/mpd.slice*.json --output=data/dictionaries

A copy has been uploaded to weights and biases

wandb artifact put -n "recsys-spotify/dictionaries" -d "Dictionaries for tracks, arists and albums" -t "dictionaries" dictionaries


