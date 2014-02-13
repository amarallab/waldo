'''
File: data_settings.py
Author: Peter Winter
Description:
'''
# standard imports
import os

#set relative paths.
project_directory = os.path.dirname(os.path.realpath(__file__)) + '/../../../'
import numpy as np
import scipy.stats as stats
assert os.path.exists(project_directory), 'where is project_directory?'

cluster_settings = {
    'project_directory': project_directory,
    'qsub_directory': project_directory + 'Logistics/Cluster/BashScripts/',
    'records_directory': project_directory + 'Logistics/Cluster/Records/'
    }

logistics_settings = {
    'raw_data_dir': '/home/projects/worm_movement/Data/MWT_RawData/',
    'organization_dir': project_directory + 'Logistics/Data/',
    #'anotated_index_url': 'https://docs.google.com/spreadsheet/pub?key=0Apos6ILhRwN8dC0xbEJzNzNhRV9ULVh6NzFlX0k3dmc&output=txt',
    'anotated_index_url': 'https://docs.google.com/spreadsheet/ccc?key=0Arzc0FDONdkUdGVweC1uNUpKUkNCNE9JM3pKM2hqOGc&output=txt',
    'phoenix_data':'/home/projects/worm_movement/Data/MWT_RawData/',
    'computer_file_prefix': 'simple_index_',
    }

mongo_settings = {
    'mongo_ip': 'rio.chem-eng.northwestern.edu',
    #'mongo_ip': 'localhost',
    'mongo_port': 27017,
    'worm_db': 'worm_db',
    'blob_collection': 'blobs',
    'result_collection': 'results',
    #'user_name':'worm'
    #'user_pswrd':'wormscrawltocoffee'
    }

processing_settings = {'treat_poly_order': 4, 'treat_window_size': 13,
                       'time_poly_order': 4, 'time_window_size': 19}


eigenworm_settings= {
    'default_file': '/home/projects/worm_movement/Results/Eigenworms/feb25_pr_comps.txt',
    'default_name': 'feb25_pr_comps',
    'feb25_pr_comps': '/home/projects/worm_movement/Results/Eigenworms/feb25_pr_comps.txt'
    }

plotting_locations = {'SingleWorm': project_directory + 'SpineProcessing/Results/SingleWorms/',
                      'TraitAggregation': project_directory + 'WormProperties/Results/TraitAggregation/'}

default_mf = {'mean': np.mean,
              'std': np.std,
              '1st_q': lambda x: stats.scoreatpercentile(x, 25),
              'median': np.median,
              '3rd_q': lambda x: stats.scoreatpercentile(x, 75),
              # todo: autocorrelation
              }

import_settings = {'min_body_lengths': 2, 'min_duration': 120, 'min_size': 50}

measurment_settings = {'time_range': [400, 1e20]}

# currently these are not used for anything.
default_datatypes = [#'smooth_length',
                     'curvature_all', 'curvature_head', 'curvature_mid', 'curvature_tail',
                     'speed_perp', 'speed_perp_tail', 'speed_perp_mid', 'speed_perp_head',
                     'speed_along', 'speed_along_tail', 'speed_along_mid', 'speed_along_head',
                     'curvature_all_bl', 'curvature_head_bl', 'curvature_mid_bl', 'curvature_tail_bl',
                     'speed_perp_bl', 'speed_perp_tail_bl', 'speed_perp_mid_bl', 'speed_perp_head_bl',
                     'speed_along_bl', 'speed_along_head_bl', 'speed_along_tail_bl', 'speed_along_tail_bl',
                     'centroid_speed', 'centroid_speed_bl', 'centroid_ang_dt', 'centroid_ang_ds',
                     #'feb25_pr_comps', # measurments from Eigenworms
                     ]
