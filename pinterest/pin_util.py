#!/usr/bin/env python
# _*_ coding: utf-8 -*-
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
  Utilities for handling pinterest images.
"""

from typing import Sequence, Tuple
import os
import json

def key_to_url(key: str)-> str:
    """
    Converts a pinterest hex key into a url.
    """
    prefix = 'http://i.pinimg.com/400x/%s/%s/%s/%s.jpg'
    return prefix % (key[0:2], key[2:4], key[4:6], key)

def id_to_filename(image_dir: str, id: str) -> str:
    filename = os.path.join(
        image_dir,
        id + ".jpg")
    return filename

def is_valid_file(fname):
    return os.path.exists(fname) and os.path.getsize(fname) > 0

def get_valid_scene_product(image_dir:str, input_file: str) -> Sequence[Tuple[str, str]]:
    """
      Reads in the Shop the look json file and returns a pair of scene and matching products.
    """
    scene_product = []
    with open(input_file, "r") as f:
        data = f.readlines()
        for line in data:
            row = json.loads(line)
            scene = id_to_filename(image_dir, row["scene"])
            product = id_to_filename(image_dir, row["product"])
            if is_valid_file(scene) and is_valid_file(product):
                scene_product.append([scene, product])
    return scene_product
