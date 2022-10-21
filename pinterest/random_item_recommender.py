#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#

"""
  Generates a html file of random product recommendations from a json catalog file.
"""
import random
import json
from typing import Dict

from absl import app
from absl import flags

import pin_util

FLAGS = flags.FLAGS
_INPUT_FILE = flags.DEFINE_string("input_file", None, "Input cat json file.")
_OUTPUT_HTML = flags.DEFINE_string("output_html", None, "The output html file.")
_NUM_ITEMS = flags.DEFINE_integer("num_items", 10, "Numer of items to recommend.")

# Required flag.
flags.mark_flag_as_required("input_file")
flags.mark_flag_as_required("output_html")

def read_catalog(catalog: str) -> Dict[str, str]:
    """
      Reads in the product to category catalog.
    """
    with open(catalog, "r") as f:
        data = f.read()
    result = json.loads(data)
    return result

def dump_html(subset, output_html:str) -> None:
    """
      Dumps a subset of items.
    """
    with open(output_html, "w") as f:
        f.write("<HTML>\n")
        f.write("""
        <TABLE><tr>
        <th>Key</th>
        <th>Category</th>
        <th>Image</th>
        </tr>""")
        for item in subset:
            key, category = item
            url = pin_util.key_to_url(key)
            img_url = "<img src=\"%s\">" % url
            out = "<tr><td>%s</td><td>%s</td><td>%s</td></tr>\n" % (key, category, img_url)
            f.write(out)
        f.write("</TABLE></HTML>")

def main(argv):
    """Main function."""
    del argv  # Unused.

    catalog = read_catalog(_INPUT_FILE.value)
    catalog = list(catalog.items())
    random.shuffle(catalog)
    dump_html(catalog[:_NUM_ITEMS.value], _OUTPUT_HTML.value)


if __name__ == "__main__":
    app.run(main)
