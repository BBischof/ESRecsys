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
  Generates a html file of random product recommendations from a json catalog file.
"""

import random
import json
import os
from typing import Sequence, Tuple

from absl import app
from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
import numpy as np

import input_pipeline

FLAGS = flags.FLAGS
_INPUT_FILE = flags.DEFINE_string("input_file", None, "Input cat json file.")
_IMAGE_DIRECTORY = flags.DEFINE_string(
    "image_dir",
    None,
    "Directory containing downloaded images from the shop the look dataset.")
_NUM_NEG = flags.DEFINE_integer(
    "num_neg", 5, "How many negatives per positive."
)

# Required flag.
flags.mark_flag_as_required("input_file")
flags.mark_flag_as_required("image_dir")

def id_to_filename(id: str) -> str:
    filename = os.path.join(
        _IMAGE_DIRECTORY.value,
        id + ".jpg")
    return filename

def get_valid_scene_product(input_file: str) -> Sequence[Tuple[str, str]]:
    """
      Reads in the Shop the look json file and returns a pair of scene and matching products.
    """
    scene_product = []
    with open(input_file, "r") as f:
        data = f.readlines()
        for line in data:
            row = json.loads(line)
            scene = id_to_filename(row["scene"])
            product = id_to_filename(row["product"])
            if os.path.exists(scene) and os.path.exists(product):
                scene_product.append([scene, product])
    return scene_product

def generate_triplets(
    scene_product: Sequence[Tuple[str, str]],
    num_neg: int) -> Sequence[Tuple[str, str, str]]:
    """Generate positive and negative triplets."""
    count = len(scene_product)
    train = []
    test = []
    for i in range(count):
        scene, pos = scene_product[i]
        is_train = i % 10 != 0
        for j in range(num_neg):
            neg_idx = random.randint(0, count - 1)
            _, neg = scene_product[neg_idx]
            if is_train:
                train.append((scene, pos, neg))
            else:
                test.append((scene, pos, neg))
    return np.array(train), np.array(test)
        
def main(argv):
    """Main function."""
    del argv  # Unused.

    scene_product = get_valid_scene_product(_INPUT_FILE.value)
    logging.info("Found %d valid scene product pairs." % len(scene_product))

    train, test = generate_triplets(scene_product, _NUM_NEG.value)
    logging.info("Train triplets\n%s" % train[0:_NUM_NEG.value])
    logging.info("Test triplets\n%s" % test[0:_NUM_NEG.value])

    train_ds = input_pipeline.create_dataset(train)
    test_ds = input_pipeline.create_dataset(test)

    for x in train_ds:
        print(x)
        print(x[0].shape, x[1].shape, x[2].shape)
        break

if __name__ == "__main__":
    app.run(main)
