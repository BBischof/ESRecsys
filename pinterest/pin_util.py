#!/usr/bin/env python
# _*_ coding: utf-8 -*-
# 
# 

"""
  Utilities for handling pinterest images.
"""

def key_to_url(key: str)-> str:
    """
    Converts a pinterest hex key into a url.
    """
    prefix = 'http://i.pinimg.com/400x/%s/%s/%s/%s.jpg'
    return prefix % (key[0:2], key[2:4], key[4:6], key)