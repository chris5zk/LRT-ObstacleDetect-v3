# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 02:26:44 2022

@author: chrischris
"""

#### For training and testing

from imppack import *

class Rail_dataset(Dataset):
    def __init__(self,
                 root,
                 ):
        