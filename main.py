from __future__ import print_function
from __future__ import division
import os
import math
import numpy as np
import cv2
from random import randint
import ast
import os
import json
import numpy as np
from shutil import copyfile
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
import cv2
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from pycocotools.coco import COCO




import os
import json
import numpy as np
from shutil import copyfile
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from pycocotools.coco import COCO
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


def load_image(path):
  im=Image.open(path)
  return im

def read_csv(csv_path):
  csv=open(csv_path,'r')
  lines=csv.readlines()
  count=0
  for line in lines[1:]:
    # print(count)
    count+=1
    words=line.split(',')
    file_name=words[0]
    print(count, file_name)
    lists=[[],[],[],[]]
    if('"' in line):
      temp=line.split('"')[1:-2:2]
      for i in range(4):
        temps=temp[i].split(', ')
        for x in temps:
          lists[i].append(int(x))
    elif(',,,,,' in line):
      continue
    else:
      for i in range(4):
        lists[i].append(int(words[i+1]))
    xmin=lists[0]
    ymin=lists[1]
    xmax=lists[2]
    ymax=lists[3]
    im=cv2.imread(file_name)
    x=im.shape[0]/10
    y=im.shape[1]/10
    size=(x,y)
    new_im=perturb(xmin,xmax,ymin,ymax,im,size)


    file_name=file_name[9:]
    cv2.imwrite('train_v3/'+file_name,new_im)

    
def perturb(xmin,xmax,ymin,ymax,image,size):
  x,y=size
  x=int(math.floor(x))
  y=int(math.floor(y))
  n_ships=len(xmin)
  xmin_p,ymin_p=get_random(xmin,xmax,ymin,ymax,image,size)
  if(xmin==None):
    return None
  image = noisy(image, xmin_p, xmin_p+x, ymin_p, ymin_p+y)
  #image[xmin_p:xmin_p+x,ymin_p:ymin_p+y,:]=np.zeros((x,y,3))
  return image

def get_random(xmin,xmax,ymin,ymax,image,size):
  x,y=size
  x=int(math.floor(x))
  y=int(math.floor(y))
  X,Y,_=image.shape
  new_image=np.zeros((X,Y))
  for i in range(len(xmin)):
    xm=xmin[i]
    xM=xmax[i]
    ym=ymin[i]
    yM=ymax[i]
    new_image[xm:xM,ym:yM]=np.ones((xM-xm,yM-ym))
  temp=np.zeros((x,y))
  for k in range(500):
    i=randint(0,X-x-1)
    j=randint(0,Y-y-1)
    if(np.array_equal(new_image[i:i+x,j:j+y],temp)):
      return i,j
  # for i in range(X-x):
  #   for j in range(Y-y):
  #     if(np.array_equal(new_image[i:i+x,j:j+y],temp)):
  #       return i,j
  return None,None

def noisy(image, x1,x2,y1,y2):
   
	s_vs_p = 0.5
	amount = 0.2
	out = np.copy(image)
      # Salt mode
	num_salt = np.ceil(amount * (x2-x1) * (y2-y1) * s_vs_p)
	coords = [np.random.randint(j, i - 1, int(num_salt))
	        for j,i in [[x1,x2],[y1,y2],[0,3]]]
	out[coords] = 1
	
      # Pepper mode
	num_pepper = np.ceil(amount* (x2-x1) * (y2-y1) * (1. - s_vs_p))
	
	coords = [np.random.randint(j, i - 1, int(num_pepper))
	        for j,i in [[x1,x2],[y1,y2],[0,3]]]
	out[coords] = 0
	
	return out

read_csv('ship_val_final_new.csv')