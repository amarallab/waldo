#!/usr/bin/env python

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

'''
Filename: rsync_blobs_files.py

Description: uses rsync to move raw data files from filesystem to cluster.

more explanation on rsync located here:
http://maururu.net/2007/rsync-only-files-matching-pattern/
'''

# standard imports
import os
import sys

# path definitions
CODE_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../../'
sys.path.append(CODE_DIR)

# nonstandard imports
from conf import settings

# globals
FILESYSTEM_DIR = settings.LOGISTICS['filesystem_data']
CLUSTER_DIR = '{address}:{data_dir}'.format(address=settings.LOGISTICS['cluster_address'],
                                            data_dir=settings.LOGISTICS['cluster_data'])
FILE_TYPES = ['.blobs', '.blob', '.summary']


def sync_raw_data_files_to_cluster(source_dir=FILESYSTEM_DIR, dest_dir=CLUSTER_DIR):
    """
    uses rsync to move all raw data files to the cluster where they can be imported.

    Breif explanation of rsync options used:
    rxync by default only moves raw data files not already on cluster
    '-a' -- archive mode
            ie. recursivly transfer subdirectories
            copy/preserve simlinks, permissions, groups, modification times
    '-v' -- verbose
    '-z' -- compress files during transfer
    '--exclude "*.filetype" -- do not move files of this type.
    """
    cmd = ('rsync -avz --exclude "*.png" --exclude "*.dat" --exclude "*.avi" '
           '{dir1} {dir2}').format(dir1=source_dir, dir2=dest_dir)
    print cmd
    os.system(cmd)

if __name__ == '__main__':
    sync_raw_data_files_to_cluster()
