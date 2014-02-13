#!/usr/bin/env python

'''
Filename: update_scaling_image_lib.py

Description: This script creates a directory with images that are used to determine scaling factors.
It searches through the data directories for any images containing a specified 'search string' in their names,
moves them to a local Scaling-Factor-Images directory and renames the images to include the source camera and the ex_id.
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import glob
import os
import sys

# path definitions
CODE_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../../'
SHARED_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../'
sys.path.append(SHARED_DIR)
sys.path.append(CODE_DIR)

# nonstandard imports
from settings.local import LOGISTICS
import update_annotations

DEFAULT_DATA_DIR = LOGISTICS['filesystem_data']
DEFAULT_SAVE_DIR = LOGISTICS['scaling-factors']

def get_index_from_data(search_string):
    """
    Creates a dictionary of ex_ids (keys) and paths to images (values). Every ex_id in the dict has a name 
    names containing the search_string. Every value contains the path to the image in that directory with the shortest name.
    """
    glob_string = '{dr}*/*{search}*.{ftype}'.format(dr=DEFAULT_DATA_DIR, search=search_string, ftype='png')
    print 'Using glob to search for: {s}'.format(s=glob_string)
    potential_scaling_images = glob.glob(glob_string)
    ex_id_im_dict = {}
    for im_path in potential_scaling_images:
        ex_id, name = im_path.split('/')[-2:]
        if ex_id not in ex_id_im_dict:
            ex_id_im_dict[ex_id] = im_path
        else:
            if len(im_path) < len(ex_id_im_dict[ex_id]):
                ex_id_im_dict[ex_id] = im_path
    return ex_id_im_dict

def copy_currently_indexed_pics_back(im_dir=DEFAULT_SAVE_DIR, source_dir=DEFAULT_DATA_DIR):
    """
    Goes throught the all images in the Scaling-Factor_Images directory. Renames them back to their origional names 
    and moves them back to the origional directories. 
    """
    pic_files = glob.glob(im_dir + '*.png')
    for pic_file in pic_files:
        path, pic_name = os.path.split(pic_file)
        source = pic_name.split('_')[0]
        pic_name = pic_name.split(source +'_')[-1]
        ex_id = pic_name[:15]
        pic_name = pic_name.split(ex_id +'_')[-1]
        print source, ex_id, pic_name
        new_name = '{data_dir}{ex_id}/{pic_name}'.format(data_dir=source_dir, ex_id=ex_id, pic_name=pic_name)
        print new_name
        cmd = 'cp {source_path} {dest}'.format(source_path=pic_file, dest=new_name)
        print cmd
        os.system(cmd)

def create_local_image_index(search_string='hema', index_dir=DEFAULT_SAVE_DIR):
    """
    Moves one image from all experiments containing the search_string to the Scaling-Factor_Images directory.
    The images are renamed to also include the source camera and the ex_id.
    """
    pic_dict = get_index_from_data(search_string=search_string)
    source_dict = update_annotations.get_source_computers()
    for ex_id, im_path in pic_dict.iteritems():
        source = source_dict.get(ex_id, '?')
        path, name = os.path.split(im_path)
        #print ex_id, source, name, im_path
        new_name = '{dr}{source}_{ex_id}_{name}'.format(dr=index_dir, ex_id=ex_id, source=source, name=name)
        print new_name
        cmd = 'cp {source_path} {destination}'.format(source_path=im_path, destination=new_name)
        #print cmd
        os.system(cmd)

if __name__ == '__main__':
    create_local_image_index()
    #get_currently_indexed_files()
