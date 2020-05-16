#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 15:55:50 2020

@author: darp_lord
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def checkDir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def squareCrop(img):
    width, height=img.size
    x=0
    y=0
    dim=width
    if width==height:
        return img
    elif width<height:
        y=np.random.randint(0,height-width)
    else:
        dim=height
        x=np.random.randint(0,width-height)
    return img.crop((x, y, x+dim, y+dim))

def cropInDir(dirname, inplace=False, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if inplace:
        target_dir=os.path.abspath(dirname)
    else:
        target_dir="/".join(os.path.abspath(dirname).split("/")[:-1])+"/cropped"
        checkDir(target_dir)
    failed=[]
    for dirname, _, filenames in os.walk(dirname):
        print("In directory", dirname)
        for filename in tqdm(filenames):
            try:
                img=squareCrop(Image.open(os.path.join(dirname, filename)))
                img.save(os.path.join(target_dir, filename))
            except Exception as e:
                failed+=[filename]
                print("Failed", filename, e)
                
if __name__=="__main__":
    cropInDir(input("Directory: "))