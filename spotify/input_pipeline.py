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

import glob
import json
import os
from typing import Sequence, Tuple, Set

import tensorflow as tf

_schema = {
   "track_context": tf.io.FixedLenFeature([5], dtype=tf.int64),
   "album_context": tf.io.FixedLenFeature([5], dtype=tf.int64),
   "artist_context": tf.io.FixedLenFeature([5], dtype=tf.int64),
   "next_track": tf.io.VarLenFeature(dtype=tf.int64),
   "next_album": tf.io.VarLenFeature(dtype=tf.int64),
   "next_artist": tf.io.VarLenFeature(dtype=tf.int64),
}

def _decode_fn(record_bytes):
  result = tf.io.parse_single_example(record_bytes, _schema)
  for key in _schema.keys():
    if key.startswith("next"):
      result[key] = tf.sparse.to_dense(result[key])
  return result

def create_dataset(
    pattern: str):
    """Creates a spotify dataset.

    Args:
      pattern: glob pattern of tfrecords.
    """
    filenames = glob.glob(pattern)
    ds = tf.data.TFRecordDataset(filenames)
    ds = ds.map(_decode_fn)
    return ds

def load_dict(dictionary_path: str, name: str):
    """Loads a dictionary."""
    filename = os.path.join(dictionary_path, name)
    with open(filename, "r") as f:
        return json.load(f)

def load_all_tracks(all_tracks_file: str,
                    track_uri_dict, album_uri_dict, artist_uri_dict):
  """Loads all tracks.

  """
  with open(all_tracks_file, "r") as f:
    all_tracks_json = json.load(f)
  all_tracks_dict = {
    int(k): v for k, v in all_tracks_json.items()
  }
  all_tracks_features = {
    k: (track_uri_dict[v["track_uri"]], album_uri_dict[v["album_uri"]], artist_uri_dict[v["artist_uri"]])
    for k,v in all_tracks_dict.items()
  }
  return all_tracks_dict, all_tracks_features
  
  
