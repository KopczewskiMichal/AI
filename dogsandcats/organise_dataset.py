# organize dataset into a useful structure
from os import makedirs, listdir, remove
from shutil import move
import os
from random import seed, random

# Set random seed for reproducibility
seed(1)

# Create directories
dataset_home = './dogs-vs-cats/'
subdirs = ['train/', 'validation/']  # Add validation directory
labeldirs = ['dogs/', 'cats/']
for subdir in subdirs:
    for labldir in labeldirs:
        newdir = os.path.join(dataset_home, subdir, labldir)
        makedirs(newdir, exist_ok=True)

# Move training dataset images into subdirectories
src_directory = os.path.join(dataset_home, 'train')
for file in listdir(src_directory):
    src = os.path.join(src_directory, file)
    if random() < 0.2:  # assuming 20% split for validation
        dst_dir = 'validation/'
    else:
        dst_dir = 'train/'
    if file.startswith('cat'):
        dst = os.path.join(dataset_home, dst_dir, 'cats', file)  # Update destination path
        if os.path.isfile(src):  # Check if src is a file
            move(src, dst)  # Move file to destination
    elif file.startswith('dog'):
        dst = os.path.join(dataset_home, dst_dir, 'dogs', file)  # Update destination path
        if os.path.isfile(src):  # Check if src is a file
            move(src, dst)  # Move file to destination

# Remove empty train directory if it exists
try:
    os.rmdir(os.path.join(dataset_home, 'train'))
except OSError:
    pass
