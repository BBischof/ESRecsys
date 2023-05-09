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
from typing import Sequence, Tuple, Set

import tensorflow as tf

_schema = {
   "track_context": tf.io.FixedLenFeature([], dtype=tf.int64),
   "album_context": tf.io.FixedLenFeature([], dtype=tf.int64),
   "artist_context": tf.io.FixedLenFeature([], dtype=tf.int64),
   "next_track": tf.io.VarLenFeature(dtype=tf.int64),
   "next_album": tf.io.VarLenFeature(dtype=tf.int64),
   "next_artist": tf.io.VarLenFeature(dtype=tf.int64),
}

def _decode_fn(record_bytes):
  return tf.io.parse_single_example(record_bytes, _schema)

def create_dataset(
    pattern: str):
    """Creates a triplet dataset.

    Args:
      pattern: glob pattern of tfrecords.
    """
    filenames = glob.glob(pattern)
    ds = tf.data.TFRecordDataset(filenames)
    ds = ds.map(_decode_fn)
    return ds