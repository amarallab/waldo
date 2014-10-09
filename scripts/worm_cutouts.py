#!/usr/bin/env python
import pathcustomize
import os

from waldo.images.cutups import cutouts_for_worms

ex_id = '20130702_135704'
savedir = '/home/projects/worm_movement/Data/cuttouts/'
savedir = '/home/projects/worm_movement/Data/cutouts/'

print savedir
print os.path.isdir(savedir)
# worm_component_dict = {5: [82961,
#                            4499,
#                            5,
#                            17366,
#                            83937,
#                            4437]}
worm_component_dict = {}
node_file =  'node_components.txt'
with open(node_file, 'r') as f:
    lines = f.readlines()
for l in lines:
    comps = [int(i) for i in l.split(',')]
    worm_component_dict[comps[0]] = comps

for n in worm_component_dict:
    print n, worm_component_dict[n]

cutouts_for_worms(ex_id,
                 savedir,
                 worm_component_dict)
