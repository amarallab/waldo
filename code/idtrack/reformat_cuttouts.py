import pandas as pd
import os
import sys
from glob import glob
index = pd.read_csv('20130702_135704/index.csv')
groups = index.groupby('bid')


wormdirs = [ 'worm_16735',
             'worm_16854',
             'worm_16855',
             'worm_13',
             'worm_13260',
             'worm_1295',
             'worm_13286',
             'worm_10470',
             'worm_10477',
             'worm_4254']
#for wd in wormdirs:
#    os.mkdir(wd)

worms = [int(w.split('_')[-1]) for w in wormdirs]

#for w, df in groups:
#    if int(w) in worms:
#        #frames = df['frames']
#        pass

image_dirs =glob('./20130702_135704/*')
print image_dirs[:10]


for w, wd in zip(worms, wormdirs):
    print w, wd
    worm_images = glob('20130702_135704/*/{w}_*.png'.format(w=w))
    for wi in worm_images:
        frame = wi.split('/')[1]
        itype = wi.split('_')[-1].split('.png')[0]
        #print wi, frame, itype

        cmd = 'cp {wi} {wd}/{frame}_{it}.png'.format(wi=wi, wd=wd,
                                                     frame=frame, it=itype)
        print cmd
        os.system(cmd)

#print frames
