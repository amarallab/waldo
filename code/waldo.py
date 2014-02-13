#!/usr/bin/env python

'''
Filename: waldo.py
Description: provides a command line user interface with which to process data.
'''
__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import argparse
import time
import json
import cProfile as profile

# path definitions
PROJECT_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../'
CODE_DIR = PROJECT_DIR + 'code/'
SHARED_DIR = CODE_DIR + 'shared/'
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

# nonstandard imports
from worms.measurement_suite import measure_all_for_ex_id
from ExportData.export_percentiles import export_blob_percentiles_by_ex_id
#from importing.import_rawdata_into_db import import_ex_id
#from importing.process_spines import process_ex_id
from importing.process_spines import process_ex_id2 as process_ex_id
import database.mongo_support_functions as mongo
from settings.local import MONGO as mongo_settings

def run_function_for_ex_ids(f, name, ex_ids):
    """ repeatedly runs a function on each ex_id in a list.
    For each run, it stores timing data and a profiler binary file.

    :param f: the function that will be run with ex_id as the only input
    :param name: a string specifying the type of job being performed
    :param ex_ids: a list of ex_ids that will be used as inputs for function, f
    """
    # initialize file names for timing and profileing
    now_string = time.ctime().replace('  ', '_').replace(' ', '_').replace(':', '.').strip()
    base_dir = '{dir}/Logistics/Cluster/'.format(dir=project_directory)
    time_file = '{dir}Timing/{type}_{now}_x{N}.json'.format(dir=base_dir, type=name, now=now_string, N=len(ex_ids))
    try:
        # initialize the connection to the mongo client
        mongo_client, _ = mongo.start_mongo_client(mongo_settings['mongo_ip'], mongo_settings['mongo_port'],
                                                   mongo_settings['worm_db'], mongo_settings['blob_collection'])
        time_storage = {}
        for ex_id in ex_ids:
            print '{function}ing {ei} starting at: {t}'.format(function=name, ei=ex_id, t=time.clock())
            profile_file = '{dir}Profile/{type}_{id}_{now}.profile'.format(dir=base_dir, type=name, now=now_string, id=ex_id)
            time_storage[ex_id] = {'start': time.clock()}
            profile.runctx('f(ex_id, mongo_client=mongo_client)', globals(), locals(), filename=profile_file)
            time_storage[ex_id]['finish'] = time.clock()

        json.dump(time_storage, open(time_file, 'w'))

    except Exception as e:
        print 'Error with {name} at time {t}\n{err}'.format(name=name, t=time.clock(), err=e)
    finally:
        mongo_client.close()

def main(args):
    """ all arguments are parsed here and the appropriate functions are called.
    :param args: arguments from argparse (namespace object)
    """
    if args.all:
        args.i, args.p, args.m = True, True, True
    if args.i:
        run_function_for_ex_ids(f=import_ex_id, name='import', ex_ids=args.ex_ids)
    if args.p:
        run_function_for_ex_ids(f=process_ex_id, name='process', ex_ids=args.ex_ids)
    if args.m:
        run_function_for_ex_ids(f=measure_all_for_ex_id, name='measure', ex_ids=args.ex_ids)
    if args.e:
        run_function_for_ex_ids(f=export_blob_percentiles_by_ex_id, name='export', ex_ids=args.ex_ids)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prefix_chars='-',
                                     description="by default it does nothing. but you can specify if it should import, "
                                                 "processes, or aggregate your data.")
    parser.add_argument('ex_ids', metavar='N', type=str, nargs='+', help='an integer for the accumulator')
    parser.add_argument('-i', action='store_true', help='import data')
    parser.add_argument('-p', action='store_true', help='process data')
    parser.add_argument('-m', action='store_true', help='measurement')
    parser.add_argument('-e', action='store_true', help='export percentiles')
    parser.add_argument('-t', action='store_true', help='records processing time')
    parser.add_argument('--all', action='store_true', help='import, process, and aggregate measurements')
    #parser.add_argument('--overwrite', action='store_true', help='overwrite previous documents in database')
    main(args=parser.parse_args())


