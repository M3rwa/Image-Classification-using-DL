import torch
from torch import nn, optim
from torchvision import transforms, datasets, models
from workspace_utils import keep_awake, active_session
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
import argparse
import utilities
args = argparse.ArgumentParser(description='predict.py')
args.add_argument('image_path', action="store", type = str, default='flowers/train/102/image_08007.jpg', 
                  help='image path for testing')
args.add_argument('path', nargs='*', action="store",type = str, default='./checkpoint.pth', 
                  help='trained model path for loading')
args.add_argument('--gpu_or_cpu', dest="gpu_or_cpu", action="store_true", default="cpu", 
                  help='use gpu or cpu for testing')
args.add_argument('--topk', dest="topk", action="store", type=int,  default=5, 
                  help='top K most likely classes')
args.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json', 
                  help='mapping from classes numbers to real names')
args = args.parse_args()
path = args.path
image_path = args.image_path
gpu_or_cpu = args.gpu_or_cpu
topk = args.topk
category_names = args.category_names
with open(category_names, 'r') as f:
    class2name = json.load(f)
model = utilities.load(path)
props, classes = utilities.predict(image_path, model, topk, gpu_or_cpu)
props = props[0].detach().numpy()
labels = [cat_to_name[str(n+1)] for n in idx]
data = {'Class':labels, 'Probablity': props}
df = pd.DataFrame(data)
print(df.to_string(index=False))