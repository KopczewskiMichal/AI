# Dataset ze strony:    https://www.kaggle.com/datasets/sudarshanvaidya/random-images-for-face-emotion-recognition
from os import makedirs, listdir, remove
from shutil import move
import os
from random import seed, random

seed(1)
dataset_home = './Dataset/'
subdirs = ['train/', 'validation/']  
labeldirs = ['anger/', 'contempt/', 'disgust/', 'fear/', 'happiness/', 'neutrality/', 'sadness/', 'surprise/']
for subdir in subdirs:
    for labldir in labeldirs:
        newdir = os.path.join(dataset_home, subdir, labldir)
        makedirs(newdir, exist_ok=True)


src_directory = os.path.join(dataset_home)
for classFolder in labeldirs:
  for file in listdir(os.path.join(src_directory,classFolder)):
      src = os.path.join(src_directory, classFolder,file)
      if random() < 0.2:
          dst_dir = 'validation/'
      else:
          dst_dir = 'train/'

      imageNumber = len(listdir(os.path.join(dataset_home, dst_dir, classFolder)))
      dst = os.path.join(dataset_home, dst_dir, classFolder, f"{imageNumber}.png")  
      if os.path.isfile(src): 
          move(src, dst)  
      

try:
    os.rmdir(os.path.join(dataset_home, 'train'))
except OSError:
    pass
