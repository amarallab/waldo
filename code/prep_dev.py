import setpath
#import images.worm_finder as wf
import collider.prep.prepare as prep

ex_ids = ['20130614_120518',
          '20130318_131111',
          '20130414_140704', # giant component(?)
          '20130702_135704', # many pics
          '20130702_135652']

#ex_id = '20130614_120518'
ex_id = '20130318_131111'

#b, t, s = prep.summarize(ex_id)
for ex_id in ex_ids:
    g = prep.graph_cache(ex_id)
#r = prep.check_roi(ex_id)
#bl = prep.bodylengths_moved(ex_id)
#pf = fm.PrepData(ex_id)
#prep.quick_check(ex_id)
