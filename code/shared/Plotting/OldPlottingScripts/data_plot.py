import os
import sys
code_directory =  os.path.dirname(os.path.realpath(__file__)) + '/../'
assert os.path.exists(code_directory), 'code directory not found'
sys.path.append(code_directory)
from Shared.Code.Settings.data_settings import settings
from glob import glob

org_dir = settings['organization_dir']
print org_dir
data_dir = settings['raw_data_dir']
print data_dir
data_dirs = glob(data_dir + '201*')


def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size
print get_size()

an_index_files = glob(org_dir + '*AnIndex*')

days_per_month = {'01':31.0,
                  '02':28.0,
                  '03':31.0,
                  '04':30.0,
                  '05':31.0,
                  '06':30.0,
                  '07':31.0,
                  '08':31.0,
                  '09':30.0,
                  '10':31.0,
                  '11':30.0,
                  '12':31.0,
                      }

data_size = []
for d_dir in data_dirs: 
    date = d_dir.split('/')[-1].split('_')[0]

    year = date[:4]
    month = date[4:6]
    day = date[6:]
    position = int(month) + (int(day) / days_per_month[month])
    data_size.append((position, get_size(d_dir)))
    
data = []


'''
for index_file in an_index_files:
    f = open(index_file, 'r')
    lines = f.readlines()
    f.close()
    print index_file

    for line in lines[1:]:
        cols = line.split()
        date = cols[0]
        year = date[:4]
        month = date[4:6]
        day = date[6:]
        position = int(month) + (int(day) / days_per_month[month])
        aprox_worm_num = cols[4]
        
        #print year, month,day, position, aprox_worm_num
        
        try: 
            #total_worms += int(aprox_worm_num)
            data.append((position,int(aprox_worm_num)))
        except: pass
'''
total_worms = 0
data = sorted(data_size)
t_data = []
for pos, worm_num in data:
    total_worms += worm_num/10.0**9
    t_data.append((pos, total_worms))

x, y = zip(*t_data)
print total_worms
from pylab import *
figure()
plot(x,y)
ylabel('total gigabytes of data collected')
xlabel('month')
show()

