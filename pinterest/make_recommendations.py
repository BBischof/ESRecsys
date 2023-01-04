#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#
# Copyright 2022 Hector Yee, Bryan Bischoff
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
  Given embedding files makes recommendations.
"""

import json
import os
from typing import Any, Dict, Tuple
import re

from absl import app
from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

import pin_util

FLAGS = flags.FLAGS
_PRODUCT_EMBED_ = flags.DEFINE_string("product_embed", None, "Product embedding json.")
_SCENE_EMBED_ = flags.DEFINE_string("scene_embed", None, "Scene embedding json.")
_TOP_K = flags.DEFINE_integer("top_k", 10, "Number of top scoring products to return per scene.")
_OUTPUT_DIR = flags.DEFINE_string("output_dir", "/tmp", "Location to write output.")
_MAX_RESULTS = flags.DEFINE_integer("max_results", 100, "Max scenes to score.")

# Required flag.
flags.mark_flag_as_required("product_embed")
flags.mark_flag_as_required("scene_embed")

def find_top_k(
  scene_embedding,
  product_embeddings,
  k):
  """
  Finds the top K nearest product embeddings to the scene embedding.

  Args:
    scene_embedding: embedding vector for the scene
    product_embedding: embedding vectors for the products.
    k: number of top results to return.
  """

  scores = scene_embedding * product_embeddings
  scores = jnp.sum(scores, axis=-1)
  scores_and_indices = jax.lax.top_k(scores, k)
  return scores_and_indices

def local_file_to_pin_url(filename):
  """Converts a local filename to a pinterest url."""
  key = filename.split("/")[-1]
  key = key.split(".")[0]
  url = pin_util.key_to_url(key)
  result = "<img src=\"%s\">" % url
  return result

def save_results(
  filename: str,
  scene_key: str,
  scores_and_indices: Tuple[Any, Any],
  index_to_key: Dict[int, str]):
  """
  Save results of a scoring run as a html document.

  Args:
    filename: name of file to save as.
    scene_key: Scene key.
    scores_and_indices: A tuple of (scores, indices).
    index_to_key: A dictionary of index to product key.
  """
  scores, indices = scores_and_indices
  scores = np.array(scores)
  indices = np.array(indices)
  with open(filename, "w") as f:
    f.write("<HTML>\n")
    scene_img = local_file_to_pin_url(scene_key)
    f.write("Nearest neighbors to %s<br>\n" % scene_img)
    for i in range(scores.shape[0]):
      idx = indices[i]
      product_key = index_to_key[idx]
      product_img = local_file_to_pin_url(product_key)
      f.write("Rank %d Score %f<br>%s<br>\n" % (i + 1, scores[i], product_img))
    f.write("</HTML>\n")

def main(argv):
    """Main function."""
    del argv  # Unused.

    tf.config.set_visible_devices([], 'GPU')
    tf.compat.v1.enable_eager_execution()
    with open(_PRODUCT_EMBED_.value, "r") as f:
      product_dict = json.load(f)
    with open(_SCENE_EMBED_.value, "r") as f:
      scene_dict = json.load(f)

    # Make the a key to embedding id and product embedding matrix.
    index_to_key = {}
    product_embeddings = []
    for index, kv in enumerate(product_dict.items()):
      key, vec = kv
      index_to_key.update({index : key})
      product_embeddings.append(np.array(vec))
    product_embeddings = jnp.stack(product_embeddings, axis=0)

    top_k_finder = jax.jit(find_top_k, static_argnames=["k"])

    for index, kv in enumerate(scene_dict.items()):
      scene_key, scene_vec = kv
      scene_embed = jnp.expand_dims(jnp.array(scene_vec), axis=0)
      scores_and_indices = top_k_finder(scene_embed, product_embeddings, _TOP_K.value)
      filename = os.path.join(_OUTPUT_DIR.value, "%05d.html" % index)
      save_results(filename, scene_key, scores_and_indices, index_to_key)
      if index > _MAX_RESULTS.value:
        break


if __name__ == "__main__":
    app.run(main)
