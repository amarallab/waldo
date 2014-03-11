#!/usr/bin/env python

'''
Filename: qsub_waldo.py
Description: runs many parallel copies of a python script on the cluster.
'''
__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import glob
import argparse

# path definitions
CODE_DIR = os.path.dirname(os.path.realpath(__file__))
SHARED_DIR = CODE_DIR + '/shared/'
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

# nonstandard imports
from database.mongo_retrieve import mongo_query
from settings.local import LOGISTICS
from annotation.experiment_index import Experiment_Attribute_Index
from wio.file_manager import ensure_dir_exists, get_ex_ids_in_worms

def qsub_run_script(python_script='waldo.py', args='', job_name='job',
                    number_of_jobs=25):
    """ Runs parallel python jobs on cluster after receiving a list of arguments for that script.

    :param python_script: which python script should be run
    :param args: the arguments that should be passed to the scripts    
    :param job_name: identifier for job type. used for output file naming.
    :param number_of_jobs: number of jobs
    """

    qsub_directory = LOGISTICS['qsub_directory']
    ensure_dir_exists(qsub_directory)
    for job_num in range(number_of_jobs):
        cmd = 'python {dir}/{py}'.format(dir=CODE_DIR, py=python_script)
        for i, ex_id in enumerate(args):
            if i%number_of_jobs == job_num:
                cmd += ' ' + str(ex_id)
        job_id = job_name + '_' + str(job_num)
        qsub_filename = qsub_directory + job_id + '.sh'
        print qsub_filename
        print cmd
        #write_sub_file(filename=qsub_filename, job_id=job_id, qsub_out_dir=qsub_directory, command_line=cmd)
        with open(qsub_filename, "w") as f:
            f.write("#! /bin/bash\n")
            f.write("#PBS -d .\n")
            f.write('#PBS -e {d}{id}-std.err\n'.format(d=qsub_directory, id=job_id))
            f.write('#PBS -o {d}{id}-std.out\n'.format(d=qsub_directory, id=job_id))
            f.write('#PBS -N {id}\n'.format(id=job_id))
            f.write("#PBS -q low\n\n\n")
            f.write(cmd)
            f.close()
        os.system('qsub '+ qsub_filename)

def list_ex_ids_with_raw_data(inventory_directory):
    ''' make list of ex_ids present in the data directory on the cluster.

    :param inventory_directory: directory to search for ex_id data.
    '''
    search_path = inventory_directory + '*'
    ex_ids = []

    for entry in glob.glob(search_path):
        dirname = entry.split('/')[-1]
        if os.path.isdir(entry) and len(dirname) == 15:
            ex_ids.append(dirname)

    if len(ex_ids) < 5:
        print 'Warning: not many ex_id directories found'
        print 'search path for raw data is ({sp})'.format(sp=search_path)
        print '{N} ex_ids present'.format(N=len(ex_ids))
    return ex_ids

def list_ex_ids_with_exported_data(export_directory):
    ''' make list of ex_ids present in the data directory on the cluster.

    :param inventory_directory: directory to search for ex_id data.
    '''
    search_string = '{dir}blob_percentiles_*.json'.format(dir=export_directory)
    print search_string
    ex_ids = []
    for entry in glob.glob(search_string):
        ex_id = entry.split(export_directory + 'blob_percentiles_')[-1][:15]
        # one minor check to see if ex_id has a _ at right location.
        if ex_id[8] == '_':
            ex_ids.append(ex_id)
    return ex_ids

def choose_ex_ids(db_attribute=('purpose', 'N2_aging'), blobfiles=None, stage1=None, stage2=None, exported=None, **kwargs):
    """ Return a list of ex_ids that match desired criteria. Good for bulk processing.

    :param db_attribute: metadata key:value pair to distinguish which experiments to process
    :param blobfiles: toggle whether raw data files should or should not be present to include ex_id
    :param stage1: toggle whether it should or should not be present in DB to include ex_id
    :param stage2: toggle whether smoothed spines should or should not be present in DB to include ex_id
    :return: list of ex_ids that match criterion
    """
    # always check which ex_ids match metadata.
    key, value = db_attribute
    ei = Experiment_Attribute_Index()
    target_ex_ids = set(ei.return_ex_ids_with_attribute(key_attribute=key, attribute_value=value))
    print '{N} ex_ids have metadata matching ({Y}:{Z})'.format(N=len(target_ex_ids), Y=db_attribute[0], Z=db_attribute[1])

    # all of these calls use an outdated model (checking mongo database) as to see if an ex_id has been processed or not.

    # Check whether or not raw data directory is present for ex_ids and modify list of target ex_ids
    if blobfiles is not None:
        ex_ids_with_blobfiles = set(list_ex_ids_with_raw_data(inventory_directory=LOGISTICS['cluster_data']))
        if blobfiles:
            target_ex_ids = target_ex_ids & ex_ids_with_blobfiles
        else:
            target_ex_ids = target_ex_ids - ex_ids_with_blobfiles
        print '{N} ex_ids have available blob files. {Y} considered further.'.format(N=len(ex_ids_with_blobfiles),
                                                                                     Y=len(target_ex_ids))

    # Check whether or not ex_ids are already in database and modify list of target ex_ids
    if stage1 is not None:
        #ex_ids_in_db = set([ex_id for ex_id in target_ex_ids
        #                    if mongo_query({'ex_id': ex_id}, {'ex_id':1}, find_one=True, **kwargs)])
        plates_with_worms = set(get_ex_ids_in_worms())
        if stage1:
            target_ex_ids = target_ex_ids & plates_with_worms
        else:
            target_ex_ids = target_ex_ids - plates_with_worms
        print 'Of these, {N} ex_ids are in the database. {Y} considered further.'.format(N=len(plates_with_worms),
                                                                                         Y=len(target_ex_ids))

    print '{a} ex_ids available for job'.format(a=len(target_ex_ids))
    return list(target_ex_ids)

def main(args, db_attribute):
    """
    Main function that interprets input arguments and activates appropriate scripts.

    :param args: arguments from command line.
    :param db_attribute: tuple to toggle behavior of script.
    """
    dataset = str(db_attribute[1])
    if args.w:
        ex_ids = choose_ex_ids(db_attribute=db_attribute, stage1=False)
        name = '{ds}-w'.format(ds=dataset)
        qsub_run_script(python_script='waldo.py -tw', job_name=name, args=ex_ids, number_of_jobs=20)
    if args.p:
        ex_ids = choose_ex_ids(db_attribute=db_attribute, stage1=False)
        name = '{ds}-p'.format(ds=dataset)
        qsub_run_script(python_script='waldo.py -tp', job_name=name, args=ex_ids, number_of_jobs=20)

    #if args.e:
    #    print 'batch export'
    #    ex_ids = choose_ex_ids(db_attribute=db_attribute, stage2=True, exported=False)
    #    qsub_run_script(python_script='waldo.py -e', job_name='export', args=ex_ids, number_of_jobs=20)

if __name__ == '__main__':
    # Toggles
    #db_attribute = ('dataset', 'N2_aging')
    #db_attribute = ('dataset', 'disease_models')
    db_attribute = ('dataset', 'thermo_recovery')
    #db_attribute = ('dataset', 'zoom_test')
    #db_attribute = ('dataset', 'copas_TJ3001')

    parser = argparse.ArgumentParser(prefix_chars='-',
                                     description='by default it does nothing. but you can specify if it should import, '
                                                 'processes, or aggregate your data.')
    parser.add_argument('-w', action='store_true', help='worms')
    parser.add_argument('-p', action='store_true', help='plates')
    parser.add_argument('-s', action='store_true', help='datasets')
    parser.add_argument('-o', action='store_true', help='overwrite')
    main(args=parser.parse_args(), db_attribute=db_attribute)



