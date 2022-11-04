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

FLAGS = flags.FLAGS
_INPUT_FILE = flags.DEFINE_string("input_file", None, "Input cat json file.")
_IMAGE_DIRECTORY = flags.DEFINE_string(
    "image_dir",
    None,
    "Directory containing downloaded images from the shop the look dataset.")

# Required flag.
flags.mark_flag_as_required("input_file")
flags.mark_flag_as_required("image_dir")

def id_exists(id: str) -> bool:
    """
    Check if we have the id in the local image directory.
    """
    filename = os.path.join(
        _IMAGE_DIRECTORY.value,
        id + ".jpg")
    return os.path.exists(filename)

def get_valid_scene_product(input_file: str) -> Sequence[Tuple[str, str]]:
    """
      Reads in the Shop the look json file and returns a pair of scene and matching products.
    """
    scene_product = []
    with open(input_file, "r") as f:
        data = f.readlines()
        for line in data:
            row = json.loads(line)
            scene_id = row["scene"]
            product_id = row["product"]
            if id_exists(scene_id) and id_exists(product_id):
                scene_product.append([scene_id, product_id])
    return scene_product

def main(argv):
    """Main function."""
    del argv  # Unused.

    scene_product = get_valid_scene_product(_INPUT_FILE.value)
    logging.info("Found %d valid scene product pairs." % len(scene_product))


if __name__ == "__main__":
    app.run(main)
