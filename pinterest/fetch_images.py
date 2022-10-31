#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#

"""
  Fetches images from pinterest to a local directory.
"""
import time
import json
from typing import FrozenSet
import os
import urllib.request
from xmlrpc.client import boolean

from absl import app
from absl import flags

import pin_util

FLAGS = flags.FLAGS
_INPUT_FILE = flags.DEFINE_string("input_file", None, "Input json file.")
_MAX_LINES = flags.DEFINE_integer("max_lines", 100000, "Max lines to read")
_SLEEP_TIME = flags.DEFINE_float("sleep_time", 10, "Sleep time in seconds.")
_SLEEP_COUNT = flags.DEFINE_integer("sleep_count", 25, "Sleep every this number of files")
_OUTPUT_DIR = flags.DEFINE_string("output_dir", None, "The output directory.")

# Required flag.
flags.mark_flag_as_required("input_file")
flags.mark_flag_as_required("output_dir")

def get_keys(input_file: str, max_lines: int) -> FrozenSet[str]:
    """
      Reads in the Shop the look json file and returns a set of keys.
    """
    keys = []
    with open(input_file, "r") as f:
        data = f.readlines()
        count = 0
        for line in data:
            if count > max_lines:
                break
            count = count + 1
            row = json.loads(line)
            keys.append(row["product"])
            keys.append(row["scene"])
    result = frozenset(keys)
    return result

def fetch_image(key: str, output_dir: str) -> boolean:
    """Fetches an image from pinterest."""
    output_name = os.path.join(output_dir, "%s.jpg" % key)
    if os.path.exists(output_name):
        print("%s already downloaded." % key)
        return False
    url = pin_util.key_to_url(key)
    got_something = False
    sleep_time = _SLEEP_TIME.value
    while not got_something:
        try:
            with urllib.request.urlopen(url) as response:
                with open(output_name, "wb") as f:
                    f.write(response.read())
                    got_something = True
        except:
            print("Network error, sleeping and retrying")
            time.sleep(sleep_time)
            sleep_time = sleep_time + 1
    return True

    

def main(argv):
    """Main function."""
    del argv  # Unused.

    keys = get_keys(_INPUT_FILE.value, _MAX_LINES.value)
    total_keys = len(keys)
    keys = sorted(keys)
    print("Found %d unique images to fetch" % total_keys)
    keys = sorted(keys)
    count = 0
    timeout_count = 0
    for key in keys:
        count = count + 1
        if fetch_image(key, _OUTPUT_DIR.value):
            timeout_count = timeout_count + 1
            if timeout_count % _SLEEP_COUNT.value == 0:
              time.sleep(_SLEEP_TIME.value)
        if count % 100 == 0:
            print("Fetched %d images of %d" % (count, total_keys))

if __name__ == "__main__":
    app.run(main)
