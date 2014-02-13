'''
Author: Peter Winter
description:
some outline and spine files are too large for a single database entry
this is to split them into muliple entries and recombine them back into one dataset.
'''

key_attributes = ['data', 'part', 'data_type', 'description', 'blob_id', 'ex_id']
default_max_size = 14680064

from total_size import total_size

def split_entry_into_part_entries(big_entry, verbose=False, num_parts=None):
    '''
    accepts one entry with an unlimited length timedict 
    and returns a the same data as list of entries
    with timedicts smaller than a maximum specified length. 

    outputs:
    entries - list of entries that have sections of a split up timedict.
    the 'part' field indicates 'YofX' where Y is the entry index and X is the
    total number of parts the initial entry was divided into.

    '''
    assert type(big_entry) == dict
    assert 'data' in big_entry
    # metadata entries have no 'data'.
    #if big_entry['data_type'] == 'metadata':
    #    return [big_entry]
    if not isinstance(big_entry['data'], dict):
        return [big_entry]

    long_timedict = big_entry['data']
    entry_template = {}
    entry_template.update(big_entry)
    if 'part' in big_entry:
        del entry_template['part']
    if '_id' in big_entry:
        del entry_template['_id']
    del entry_template['data']
    

    timedict_parts, part_indicies = split_timedict_into_parts(long_timedict, number_of_parts=num_parts)
    entries = []
    for timedict_part, part_index in zip(timedict_parts, part_indicies):
        entry = {}
        entry.update(entry_template)
        entry['data'] = timedict_part
        entry['part'] = part_index
        entries.append(entry)
    return entries

def combine_part_entries_to_entry(entries):
    '''
    combines a list of entries with parts of the same dataset into one entry.
    each of fields the entries must be identical with the exception of 'data' and 'parts'

    inputs:
    entries - a list containing dictionary entries in the format of database entries

    outputs:
    big_entry - an entry with all timedict 'data' combined into one long timedict.
               the 'part' filed reads as 'allofX' where X is the number of parts combined.
    '''
    assert type(entries) in [list, tuple]
    assert len(entries) >= 1
    timedict_parts = []
    part_indicies = []
    big_entry = {}
    big_entry.update(entries[0])
    if 'part' in big_entry:
        del big_entry['part']
    if 'data' in big_entry:
        del big_entry['data']
    if '_id' in big_entry:
        del big_entry['_id']

    for entry in entries:
        assert type(entry) == dict
        # TODO: write a more specific test
        for ka in key_attributes:
            assert ka in entry, str(ka) + ' not in entry'
        timedict_parts.append(entry['data'])
        part_indicies.append(entry['part'])
        
        #make sure not 'data' sections of entries match exactly
        for atr in entry:
            if atr not in ['_id', 'data', 'part']:
                if entry[atr] != big_entry[atr]:
                    #assert entry[atr] == big_entry[atr], str(atr) + ' not matching'
                    print '#### Warning! ####'
                    print '{at} for {bid} not matching for part {p}'.format(at=atr, bid=entry['blob_id'], p=entry['part'])
                    print 'Update database so these values match: {v1}, {v2}'.format(v1=entry[atr], v2=big_entry[atr])

    long_timedict = combine_timedict_parts(timedict_parts, part_indicies)
    big_entry['data'] = long_timedict
    if len(entries) == 1:
            big_entry['part'] = '1of1'
    else:
        big_entry['part'] = 'allof' + str(len(part_indicies))
    return big_entry

def split_timedict_into_parts(long_timedict, number_of_parts=None, max_size_of_timedict=default_max_size):
    '''
    this takes a single long timedict and returns a list of multiple
    smaller timedicts and the cooresponding part indentifier.

    timedict_parts - list of timedicts containing at most the max_timepoints_in_timedict
    part_indicies - list of strings identifying which part of how many parts (for example '1of2')
    '''
    assert isinstance(long_timedict, dict)
    assert isinstance(max_size_of_timedict, int)
    size_of_timedict = total_size(long_timedict)
    if not number_of_parts:
        number_of_parts = (size_of_timedict / max_size_of_timedict + 1)
    part_indicies = ['{0}of{1}'.format(i, number_of_parts) for i in xrange(1, number_of_parts + 1)]
    step = (len(long_timedict) / number_of_parts)
    split_indicies = [i*step for i in xrange(number_of_parts)]
    #split_indicies = range(0, len(long_timedict), (len(long_timedict) / number_of_parts))
    split_indicies.append(len(long_timedict))
    #print part_indicies, split_indicies
    assert len(part_indicies) == len(split_indicies) - 1

    timedict_parts = []

    #if number_of_parts > 1: print number_of_parts
    sequential_times = sorted(long_timedict)

    for i, part_index in enumerate(part_indicies):
        part_timedict = {}
        start_index = split_indicies[i]
        end_index = split_indicies[i+1]
        for t in sequential_times[start_index:end_index]:
            part_timedict[t] = long_timedict[t]

        timedict_parts.append(part_timedict)
    return timedict_parts, part_indicies

def combine_timedict_parts(timedict_parts, part_indicies=[]):
    '''
    combines multiple timedict parts into one long timedict
    part indicies is only present to do checks on input.

    timedict_parts - list of timedicts to be combined
    part indicies - a list like this ['1ofX', '2ofX',...'XofX']
    '''
    assert type(timedict_parts) in [list, tuple]
    # this is all checks to see if part_indicies make sense.
    #print part_indicies, len(part_indicies)
    #print len(timedict_parts)

    # TODO: make more informative error messages.
    if len(part_indicies) > 0:
        assert len(part_indicies) == len(timedict_parts), 'parts error'
        part_count = part_indicies[0].split('of')[-1]
        for part in part_indicies:
            #print part
            assert part.split('of')[-1] == part_count
        #print len(part_indicies)
        #print part_count
        assert len(part_indicies) == int(part_count), str(len(part_indicies)) + str(part_count)

    long_timedict = {}
    #print 'entry lengths', [len(timedict_part) for timedict_part in timedict_parts]
    for timedict_part in timedict_parts:
        long_timedict.update(timedict_part)
    return long_timedict

def combine_entries_with_parts(all_blob_ids, all_values_for_found_entries, all_parts):
    '''
    takes three lists and returns two lists.
    TODO: rework this. return parts and make sure parts arrives. right now this is indept
    of the other scripts I wrote.
    
    inputs:
    all_blob_ids - list of blob ids
    all_values_for_found_entries - list of values to concatinate
    all_parts - list of part indicies
    '''
    # TODO: Figure out what the hell this function is good for !!!!!
    values_by_blob_id = {}
    num_parts_by_blob_id = {}
    for blob_id, timedict, part in zip(all_blob_ids, all_values_for_found_entries, all_parts):
        if blob_id not in values_by_blob_id:
            values_by_blob_id[blob_id] = {}
        values_by_blob_id[blob_id].update(timedict)
        if blob_id not in num_parts_by_blob_id:
            num_parts_by_blob_id[blob_id] = []
        num_parts_by_blob_id[blob_id].append(part)


    blob_ids, values, new_parts = [], [], []
    for blob_id in values_by_blob_id:
        # write check if all parts in numparts list
        blob_ids.append(blob_id)
        values.append(values_by_blob_id[blob_id])
        new_parts.append('allof'+str(num_parts_by_blob_id[blob_id]))

    return blob_ids, values, new_parts
        
