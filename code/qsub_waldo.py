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

# path definitions
CODE_DIR = os.path.dirname(os.path.realpath(__file__))
SHARED_DIR = CODE_DIR + '/shared/'
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

# nonstandard imports
from settings.local import LOGISTICS
from annotation.experiment_index import Experiment_Attribute_Index, list_ex_ids_with_raw_data
from wio.file_manager import ensure_dir_exists, get_ex_ids_in_worms
from waldo import create_parser

QSUB_DIR = os.path.abspath(LOGISTICS['qsub_directory'])
#QSUB_DIR = '.'

print 'saving to', QSUB_DIR

def qsub_run_script(python_script='waldo.py', args='', ex_ids=[], job_name='job', qsub_dir=QSUB_DIR,
                    number_of_jobs=25):
    """ Runs parallel python jobs on cluster after receiving a list of arguments for that script.

    :param python_script: which python script should be run
    :param args: the arguments that should be passed to the scripts    
    :param job_name: identifier for job type. used for output file naming.
    :param number_of_jobs: number of jobs
    """
 
    ensure_dir_exists(qsub_dir)
    # if there are more jobs than recordings. reduce the number of jobs.
    if len(ex_ids) < number_of_jobs:
        number_of_jobs = len(ex_ids)
                                                                   
    for job_num in range(number_of_jobs):
                
        eIDs = ex_ids[job_num::number_of_jobs]       
        job_id = '{name}_{num}'.format(name=job_name, num=job_num)
        qsub_filename = '{d}/{n}.sh'.format(d=qsub_dir, n=job_id)
        
        py = '{dir}/{py}'.format(dir=CODE_DIR, py=python_script)
        python_call = 'python2.7 {py}'.format(py=py)

        lines = ["#! /bin/bash\n",
                "#PBS -d .\n",
                '#PBS -e {d}/{ID}-std.err\n'.format(d=qsub_dir, ID=job_id),
                '#PBS -o {d}/{ID}-std.out\n'.format(d=qsub_dir, ID=job_id),
                '#PBS -N {id}\n'.format(id=job_id),
                "#PBS -q low\n\n\n",
                '{py} {args} {IDs}'.format(py=python_call,
                                           args=args, 
                                           IDs=' '.join(eIDs))]
        
        print qsub_filename
        with open(qsub_filename, "w") as f:
            for line in lines:
                f.write(line)
            f.close()
        os.system('qsub '+ qsub_filename)


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

    # remove ex_ids from target ex_ids if already processed
    if stage1 is not None:
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
    new_args = '-c %s' % args.c if args.c is not None else ''
    dataset = str(db_attribute[1])
    
    # initiallize recording selection to have no criterion.
    blobfiles, stage1, stage2 =None, None, None
    # if not overwrite: do not process recordings with data present
    if not args.o:
        stage1 = False

    # TODO: rewrite this code so that all arguments for qsub_waldo passed on to waldo.
    if args.centroid:
        ex_ids = choose_ex_ids(db_attribute=db_attribute, stage1=stage1)
        name = '{ds}-w'.format(ds=dataset)
        qsub_run_script(python_script='waldo.py -to --centroid', job_name=name,
                        args=new_args, ex_ids=ex_ids, number_of_jobs=100)        
        return # centroid specifies that only centroid should be processed.
    if args.w:
        ex_ids = choose_ex_ids(db_attribute=db_attribute, stage1=stage1)
        name = '{ds}-w'.format(ds=dataset)
        qsub_run_script(python_script='waldo.py -tw', job_name=name,
                        args=new_args, ex_ids=ex_ids, number_of_jobs=100)
    if args.p:
        ex_ids = choose_ex_ids(db_attribute=db_attribute, stage1=True)
        name = '{ds}-p'.format(ds=dataset)
        qsub_run_script(python_script='waldo.py -tpo', job_name=name,
                        args=new_args, ex_ids=ex_ids, number_of_jobs=100)

if __name__ == '__main__':
    # Toggles
    db_attribute = ('dataset', 'N2_aging')
    db_attribute = ('dataset', 'and')
    db_attribute = ('dataset', 'disease_models')
    #db_attribute = ('dataset', 'thermo_recovery')
    #db_attribute = ('dataset', 'zoom_test')
    #db_attribute = ('dataset', 'copas_TJ3001_lifespan')
    parser = create_parser(for_qsub=True)
    main(args=parser.parse_args(), db_attribute=db_attribute)



