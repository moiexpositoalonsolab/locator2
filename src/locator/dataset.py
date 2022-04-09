import glob
import math
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset



"""
This is where you will define how you will get data to push through the neural network
all pytorch datasets need an __init__, __len__, and __getitem__ methods
"""
class Name_Of_Dataset(Dataset):
    # the init method actually creates the dataset object. Usually it will be
    # a pandas dataframe or a numpy array that stores your data. There are fancier
    # datasets you can create, but this is the most simple
    def __init__(self, vcf_path, locations_path):
        ####### Class fields ##########
        # ie:
#         self.path = DATA_PATH
#         self.data = pd.read_csv(self.path)
        # TODO: add code to properly load in vcf and locations data
        self.vcf = load(vcf_path)
        self.locations = load(locations_path)
        # TODO: add code to map index 0 - N-1 to the sample IDs
        self.index_2_id = {
                # dictionary that maps index 1-N to the sampleID
        }

    # the len method returns how many observations are in the dataset.
    # depending on the dataset, this could get fancier (ie if you're using)
    # vcf data and you have one file for a diploid organism and your individual
    # data piece is one genome, then your length would be num_vcf * 2
    def __len__(self):
        return len(self.locations)

    # the __getitem__ method describes how you return one observation from your
    # dataset. For example, if you're using genomes, then this method would go fetch
    # the vcf file off the system, load in the genome and do any data preprocessing you
    # need to get the data ready to feed into the neural network.
    # Note: the thing this method returns is the input to the neural network defined in
    # network.py called INPUT_DATA
    def __getitem__(self, idx):
        # idx is the index of the item from the dataset you want
        # for example:
        id = self.index_2_id[idx]
        # TODO: figure out how to grab
        # the row that corresponds to this sample
        # from both the vcf file and the locations file
        input = self.vcf[id]
        output = self.locations[id]
        return input, output
