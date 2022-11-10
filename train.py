import torch
from torch import nn, optim
from torchvision import transforms, datasets, models
from workspace_utils import keep_awake, active_session
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import argparse
import utilities
args = argparse.ArgumentParser(description='train.py')
args.add_argument('--arch', dest="arch", action="store", choices=["vgg16","vgg19"], default="vgg16", type=str, 
                  help='select the architecture of the network')
args.add_argument('data_dir', nargs='?', action="store", default="./flowers/", 
                  help='dataset folder for training, validation and testing')
args.add_argument('--hidden_units', dest="hidden_units", type=int, action="store", default=512, 
                  help='number of hidden units for model building')
args.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001, 
                  help='Learning Rate for the training')
args.add_argument('--gpu_or_cpu', dest="gpu_or_cpu", action="store_true", default="cpu", 
                  help='use gpu or cpu for training')
args.add_argument('--epochs', dest="epochs", action="store", type=int, default=4, 
                  help='Number of epochs for model training')
args.add_argument('--check_every', dest="check_every", action="store", type=int, default=1, 
                  help='Inspect the training and validation loss and the validatin accuracy after number of batch training')
args.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth", 
                  help='file name with its extension and path for saving the trained model')
args.add_argument('--arch_name', dest="arch_name", action="store", default="vgg16", type = str, 
                  help='model name for saving in the checkpoint')
args.add_argument('--input_size', dest="input_size", action="store", type=int, default=25088, 
                  help='model input feature for saving')
args.add_argument('--output_size', dest="output_size", action="store", type=int, default=102, 
                  help='model output feature for saving (the number of labels)')
args = args.parse_args()
data_dir = args.data_dir
arch = args.arch
hidden_units = args.hidden_units
lr = args.learning_rate
gpu_or_cpu = args.gpu_or_cpu
epochs = args.epochs
check_every = args.check_every
file_name = args.save_dir
arch_name = args.arch_name
input_size = args.input_size
output_size = args.output_size
with open('cat_to_name.json', 'r') as f:
    flower_to_name = json.load(f)
datasets, dataloaders = utilities.load_data(data_dir)
model = utilities.model(arch, hidden_units):
criterion, optimizer = utilities.compilation(model, lr)
model = utilities.fit(model, trainloader, epochs, check_every, gpu_or_cpu)
utilities.evaluate(model, testloader)
utilities.save(file_name, traindata_folder, arch_name, input_size, output_size, batch_size, epochs, optimizer)