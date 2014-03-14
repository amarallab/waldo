#!/usr/bin/env python

#TEST

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
CODE_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.path.abspath(CODE_DIR + '/../')
SHARED_DIR = CODE_DIR + '/shared/'
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

# nonstandard imports
from wio.file_manager import ensure_dir_exists
from importing.process_spines import process_ex_id
import database.mongo_support_functions as mongo
from settings.local import MONGO

TIMING_DIR = PROJECT_DIR + '/data/diagnostics/timing'
PROFILE_DIR = PROJECT_DIR + '/data/diagnostics/profileing'

MONGO_SETTINGS = {'ip':MONGO['ip'], 'port':MONGO['port'],
                  'database_name':MONGO['database'],
                  'collection_name':MONGO['blobs']}

def run_function_for_ex_ids(f, name, ex_ids, timing_dir=TIMING_DIR, no_mongo=True,
                            profile_dir=PROFILE_DIR):
    """ repeatedly runs a function on each ex_id in a list.
    For each run, it stores timing data and a profiler binary file.

    :param f: the function that will be run with ex_id as the only input
    :param name: a string specifying the type of job being performed
    :param ex_ids: a list of ex_ids that will be used as inputs for function, f
    """
    # initialize file names for timing and profileing
    now_string = time.ctime().replace(' ', '_').replace(':', '.').strip()
    ensure_dir_exists(timing_dir)
    time_file = '{dir}/{type}_{now}_x{N}.json'.format(dir=timing_dir, type=name,
                                                      now=now_string, N=len(ex_ids))

    if no_mongo:
        time_storage = {}
        ensure_dir_exists(profile_dir)        
        for ex_id in ex_ids:
            print '{fun}ing {ei} starting at: {t}'.format(fun=name, ei=ex_id,
                                                          t=time.clock())
            profile_file = '{dir}/{type}_{id}_{now}.profile'.format(dir=profile_dir,
                                                                    type=name,
                                                                    now=now_string,
                                                                    id=ex_id)
            time_storage[ex_id] = {'start': time.clock()}
            try:
                profile.runctx('f(ex_id)', globals(), locals(), filename=profile_file)
                time_storage[ex_id]['finish'] = time.clock()
                json.dump(time_storage, open(time_file, 'w'))
            except Exception as e:
                print 'Error with {name} at time {t}\n{err}'.format(name=name, t=time.clock(), err=e)
        return True

    # if mongo is being used... run this version.
    mongo_client = None
    try:
        # initialize the connection to the mongo client
        mongo_client, _ = mongo.start_mongo_client(**MONGO_SETTINGS)
        time_storage = {}
        ensure_dir_exists(profile_dir)        
        for ex_id in ex_ids:
            print '{fun}ing {ei} starting at: {t}'.format(fun=name, ei=ex_id,
                                                          t=time.clock())

            profile_file = '{dir}/{type}_{id}_{now}.profile'.format(dir=profile_dir,
                                                                    type=name,
                                                                    now=now_string,
                                                                    id=ex_id)
            time_storage[ex_id] = {'start': time.clock()}
            profile.runctx('f(ex_id, mongo_client=mongo_client)',
                           globals(), locals(), filename=profile_file)
            time_storage[ex_id]['finish'] = time.clock()

        json.dump(time_storage, open(time_file, 'w'))

    except Exception as e:
        print 'Error with {name} at time {t}\n{err}'.format(name=name, t=time.clock(), err=e)
    finally:
        if mongo_client:
            mongo_client.close()

def main(args):
    """ all arguments are parsed here and the appropriate functions are called.
    :param args: arguments from argparse (namespace object)
    """

    if args.c is not None:
        print 'Error with -c argument'

    if args.all:
        args.w, args.p, args.s = True, True, True
    if args.w:
        run_function_for_ex_ids(f=process_ex_id, name='worm', ex_ids=args.ex_ids)
    if args.p:
        run_function_for_ex_ids(f=process_ex_id, name='plate', ex_ids=args.ex_ids)
    if args.s:
        run_function_for_ex_ids(f=process_ex_id, name='dataset', ex_ids=args.ex_ids)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prefix_chars='-',
                                     description="by default it does nothing. but you can specify if it should import, "
                                                 "processes, or aggregate your data.")
    parser.add_argument('ex_ids', metavar='N', type=str, nargs='+', help='an integer for the accumulator')
    #parser.add_argument('-i', action='store_true', help='import data')
    parser.add_argument('-c', help='configuration username')
    parser.add_argument('-w', action='store_true', help='create worm data (stage 1)')
    parser.add_argument('-p', action='store_true', help='compile plate timeseries (stage 2)')
    parser.add_argument('-s', action='store_true', help='dataset processing (stage 3)')
    #parser.add_argument('-e', action='store_true', help='export percentiles')
    parser.add_argument('-t', action='store_true', help='records processing time')
    parser.add_argument('--all', action='store_true', help='import, process, and aggregate measurements')
    #parser.add_argument('--overwrite', action='store_true', help='overwrite previous documents in database')
    main(args=parser.parse_args())


