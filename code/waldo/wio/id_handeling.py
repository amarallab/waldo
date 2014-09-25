from __future__ import absolute_import

# standard library

# third party

# package specific
from . import file_manager as fm
from waldo.annotation.experiment_index import Experiment_Attribute_Index2

def is_number(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True



def classify_ids(ids):
    def is_ex_id(i):
        i = str(i)
        s = i.split('_')
        if len(s) != 2:
            return False
        if not is_number(s[0]):
            return False
        if not is_number(s[1]):
            return False
        return True

    def is_dset(i):
        ei = Experiment_Attribute_Index2()
        if i in set(ei['dataset']):
            return True
        else:
            return False


    id_types = []
    for i in ids:
        if is_ex_id(i):
            id_types.append('ex_id')
        elif is_dset(i):
            id_types.append('dset')
        else:
            id_types.append('other')

    id_types = list(set(id_types)) # remove duplicates
    if len(id_types) > 1:
        return 'mixed'
    else:
        return id_types[0]

def organize_ex_ids(ex_ids):
    eids_by_dset = {}
    for eid in ex_ids:
        dset = fm.get_dset(eid)
        if dset not in eids_by_dset:
            eids_by_dset[dset] = []
        eids_by_dset[dset].append(eid)
    return eids_by_dset

def organize_dsets(dsets):
    eids_by_dset = {}
    for dset in dsets:
        ei = Experiment_Attribute_Index2(dset)
        #ei = ei[ei['age'] == 'A1']
        #ei = ei[ei['strain'] == 'N2']
        eids_by_dset[dset] = list(ei.index)
        #print set(ei['dataset'])
        #print dset
    return eids_by_dset

def parse_ids(ids):
    id_type = classify_ids(ids)
    print 'using [{it}] as id type'.format(it=id_type)
    if id_type == 'mixed':
        print 'please only use ex_ids or dataset names, not both.'
    elif id_type == 'ex_id':
        return organize_ex_ids(ids)
    elif id_type == 'dset':
        return organize_dsets(ids)
    else:
        print 'type of id not recognized'
