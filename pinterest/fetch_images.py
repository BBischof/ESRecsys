#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#

"""
  Fetches images from pinterest to a local directory.
"""
import random
import json
from typing import FrozenSet

from absl import app
from absl import flags

import pin_util

FLAGS = flags.FLAGS
_INPUT_FILE = flags.DEFINE_string("input_file", None, "Input json file.")
_MAX_LINES = flags.DEFINE_integer("max_lines", 1000, "Max lines to read")
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


def main(argv):
    """Main function."""
    del argv  # Unused.

    keys = get_keys(_INPUT_FILE.value, _MAX_LINES.value)
    print("Found %d unique images to fetch" % len(keys))

if __name__ == "__main__":
    app.run(main)
