#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#
# Copyright 2023 Hector Yee, Bryan Bischoff
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
  Given Json playlist files makes training data.
"""

import glob
import json
import os
from typing import Any, Dict, Tuple

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

import input_pipeline

FLAGS = flags.FLAGS
_PLAYLISTS = flags.DEFINE_string("playlists", None, "Playlist json glob.")
_DICTIONARY_PATH = flags.DEFINE_string("dictionaries", "data/dictionaries", "Dictionary path.")
_OUTPUT_PATH = flags.DEFINE_string("output", "data/training", "Output path.")
_TOP_K = flags.DEFINE_integer("topk", 5, "Top K tracks to use as context.")

# Required flag.
flags.mark_flag_as_required("playlists")

def main(argv):
    """Main function."""
    del argv  # Unused.

    tf.config.set_visible_devices([], 'GPU')
    tf.compat.v1.enable_eager_execution()
    playlist_files = glob.glob(_PLAYLISTS.value)
    
    track_uri_dict = input_pipeline.load_dict(_DICTIONARY_PATH.value, "track_uri_dict.json")
    print("%d tracks loaded" % len(track_uri_dict))
    artist_uri_dict = input_pipeline.load_dict(_DICTIONARY_PATH.value, "artist_uri_dict.json")
    print("%d artists loaded" % len(artist_uri_dict))
    album_uri_dict = input_pipeline.load_dict(_DICTIONARY_PATH.value, "album_uri_dict.json")
    print("%d albums loaded" % len(album_uri_dict))
    topk = _TOP_K.value

    raw_tracks = {}

    for pidx, playlist_file in enumerate(playlist_files):
        print("Processing ", playlist_file)
        with open(playlist_file, "r") as file:
            data = json.load(file)
            playlists = data["playlists"]
            tfrecord_name = os.path.join(_OUTPUT_PATH.value, "%05d.tfrecord" % pidx)
            with tf.io.TFRecordWriter(tfrecord_name) as file_writer:
              for playlist in playlists:
                  if playlist["num_tracks"] < topk:
                      continue
                  tracks = playlist["tracks"]
                  # The first topk tracks are all for the context.
                  track_context = []
                  artist_context = []
                  album_context = []
                  # The rest are for predicting.
                  next_track = []
                  next_artist = []
                  next_album = []
                  for tidx, track in enumerate(tracks):
                      track_uri_idx = track_uri_dict[track["track_uri"]]
                      artist_uri_idx = artist_uri_dict[track["artist_uri"]]
                      album_uri_idx = album_uri_dict[track["album_uri"]]
                      if track_uri_idx not in raw_tracks:
                          raw_tracks[track_uri_idx] = track
                      if tidx < topk:
                          track_context.append(track_uri_idx)
                          artist_context.append(artist_uri_idx)
                          album_context.append(album_uri_idx)
                      else:
                          next_track.append(track_uri_idx)
                          next_artist.append(artist_uri_idx)
                          next_album.append(album_uri_idx)
                  record = tf.train.Example(
                    features=tf.train.Features(feature={
                      "track_context": tf.train.Feature(int64_list=tf.train.Int64List(value=track_context)),
                      "album_context": tf.train.Feature(int64_list=tf.train.Int64List(value=album_context)),
                      "artist_context": tf.train.Feature(int64_list=tf.train.Int64List(value=artist_context)),
                      "next_track": tf.train.Feature(int64_list=tf.train.Int64List(value=next_track)),
                      "next_album": tf.train.Feature(int64_list=tf.train.Int64List(value=next_album)),
                      "next_artist": tf.train.Feature(int64_list=tf.train.Int64List(value=next_artist)),
                    }))
                  record_bytes = record.SerializeToString()
                  file_writer.write(record_bytes)
         
    filename = os.path.join(_OUTPUT_PATH.value, "all_tracks.json")
    with open(filename, "w") as f:
        json.dump(raw_tracks, f)

if __name__ == "__main__":
    app.run(main)
