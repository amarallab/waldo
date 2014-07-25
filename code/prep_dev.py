import os
os.environ.setdefault('WALDO_SETTINGS', 'default_settings')
import setpath
from conf import settings


import images.worm_finder as wf
#import collider.prep.prepare as prep

ex_ids = ['20130614_120518',
          '20130318_131111',
          '20130414_140704', # giant component(?)
          '20130702_135704', # many pics
          '20130702_135652']

# 20130614_120518 20130318_131111 20130414_140704 20130702_135704 20130702_135652


ex_id = '20130614_120518'
#ex_id = '20130318_131111'
ex_id = '20130414_140704'

wf.draw_colors_on_image(ex_id, 30*60)
#for ex_id in ex_ids:
#b, t, s = prep.summarize(ex_id)
#print(t.head())
#g = prep.graph_cache(ex_id)
#r = prep.check_roi(ex_id)
#bl = prep.bodylengths_moved(ex_id)
#pf = fm.PrepData(ex_id)
#prep.quick_check(ex_id)
