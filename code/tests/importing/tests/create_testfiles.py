#!/usr/bin/env python

'''
Filename: 
Discription: 
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import json
import os
import sys
import glob
from itertools import izip
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes

# path specifications
test_directory = os.path.dirname(os.path.realpath(__file__)) 
project_directory = test_directory + '/../../../'
shared_directory = project_directory + 'code/shared/'
test_data_dir = test_directory + '/data/'
sys.path.append(shared_directory)
sys.path.append(project_directory + '/code/')

# nonstandard imports
from Encoding.decode_outline import encode_outline
from GeometricCalculations.matrix_and_point_operations import close_outline_border
from settings.local import SPREADSHEET, LOGISTICS

def outline_timedict_to_list(outline_timedict):
    outlines = [(float(str(k).replace('?', '.')), points)
                for (k, points) in outline_timedict.iteritems()]
    outlines.sort()
    return outlines

def format_blob_into_lines(times, outlines):

    N = len(times)
    assert N == len(outlines), 'error, json is not'

    column_names = ['im_num', 'time',
                    'x', 'y', 'size',
                    'x_coor', 'y_coor', 'orth_std',
                    'length', 'width',
                    '%', 'raw_spine',
                    '%%', 'encoded_outline']        

    column_data = {}
    for outline in outlines:
        properties = outline_points_to_properties(outline)
        for name in column_names:
            if name not in column_data:
                column_data[name] = []
            column_data[name].append(properties.get(name, '0'))

    column_data['im_num'] = map(str, range(N))
    column_data['time'] = map(lambda x: '%.3f' % x, times)
            
    '''
    # these are not appropriately valued
    column_data['x'] = map(str, range(N))
    column_data['y'] = map(str, range(N))    
    column_data['size'] = map(str, range(N))    
    # these are not appropriately valued
    column_data['x_std'] = map(str, range(N))    
    column_data['y_std'] = map(str, range(N))    
    column_data['orth_std'] = map(str, range(N))    
    # these are not appropriately valued
    column_data['width'] = map(str, range(N))    
    column_data['length'] = map(str, range(N))
    '''
    # for formatting
    column_data['%'] = ['%' for _ in xrange(N)]
    column_data['raw_spine'] = [' '.join(map(str, range(0, 66, 3))) for _ in xrange(N)]
    # the most important
    column_data['%%'] = ['%%' for _ in xrange(N)]
    encoded_outline_parts = map(encode_outline, outlines)
    column_data['encoded_outline'] = ['{x} {y} {l} {outline}'.format(x=x, y=y,
                                                                     l=l, outline=o)
                                        for ((x, y), l, o)
                                        in encoded_outline_parts]
    lines = []
    for i in range(N):
        line = ''
        for col in column_names:
            line += str(column_data[col][i]) + ' '
        lines.append(line)
    return lines

def outline_points_to_properties(outline_points):
    """
    """
    # dictionary to store all values
    properties = {}

    # make sure the outline is fully closed
    outline = close_outline_border(outline_points)
    xs, ys = zip(*outline)
    # make a blank numpy matrix just big enough to fit worm shape.
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    outline_matrix = np.zeros([x_range + 2, y_range + 2], dtype=int)
    offset_x = 1 - min(xs)
    offset_y = 1 - min(ys)
    # now fill in all the outline points
    # but move coordinates so they fit in box.
    boxed_xs = [(x + offset_x) for x in xs]
    boxed_ys = [(y + offset_y) for y in ys]
    for x, y in izip(boxed_xs, boxed_ys):
        outline_matrix[x][y] = 1
    # fill the inside of the matrix
    filled_matrix = binary_fill_holes(outline_matrix)
    
    # calculate properties
    properties['size'] = int(sum(sum(filled_matrix)))
    x, y = get_xy(filled_matrix)    
    properties['x'] = x - offset_x
    properties['y'] = y - offset_y

    # everything else involves fitting a line to the points.    
    # things not calculated:
    # xcoor, ycoor, orth_std    
    # length, width

    return properties
        
def write_blob_file_for_test_folder(ex_id='00000000_000001'):

    # make sure the write directory is all setup
    blobs_file_name = '{dir}blobfiles/{eID}/test_blobsfile_00000k.blobs'.format(dir=test_data_dir, eID=ex_id)
        
    # for test ex_id, grab all jsons that show worm outlines in point form
    test_jsons = glob.glob('data/jsons/{eID}/*.json'.format(eID=ex_id)) 
    if len(test_jsons) == 0:
        print 'no test json files found in', test_data_dir

    lines = []
    for i, jfile in enumerate(test_jsons[:], start=1):
        # get id line ready
        lines.append('% {i}'.format(i=i))
        # parse json file to get blob outline data.
        data = json.load(open(jfile, 'r'))
        outlines = outline_timedict_to_list(data)
        times, outlines_as_points = zip(*outlines)
        # make sure all points are in integer form        
        outlines_as_points = [[( int(x), int(y)) for (x, y) in o]
                              for o in outlines_as_points]
        
        # use outline data to create a set of blobs file lines
        newlines = format_blob_into_lines(times, outlines_as_points)
        map(lines.append, newlines)
        # add an empty line after blob is done.
        lines.append('')

    # write all the blobs file lines we just created.
    with open(blobs_file_name, 'w') as f:
        f.write("\n".join(lines))

def get_xy(nparray):
    N = nparray.sum()

    def getx(nparray):
        return float(sum(sum(nparray) * range(1, len(nparray[0]) + 1))) / nparray.sum()
    x = getx(nparray)
    y = len(nparray.T[0]) - getx(nparray.T) + 1
    return x, y

def test_arrays():
    '''
    x = np.array([[1,0,0],
                  [0,0,0],
                  [0,0,0]])
    print get_xy(x)
    x = np.array([[0,0,0],
                  [0,1,0],
                  [0,0,0]])
    print get_xy(x)
    x = np.array([[0,0,0],
                  [0,0,0],
                  [1,0,0]])
    print get_xy(x)
    '''
    x = np.array([[1,0],
                  [0,0],
                  [1,0]])
    print get_xy(x)
    x = np.array([[1,0, 0],
                  [0,1, 0]])
                  
    print get_xy(x)

def write_summary_file(ex_id='00000000_000001'):
    """
    1. Image number (counts up from 1)
    2. Image time (in seconds from the beginning of the experiment)
    3. Number of objects tracked
    4. Number of objects persisting
    5. Average duration for an object to have been tracked
    6. Average speed in pixels/second
    7. Average angular speed in radians/second
    8. Average length of object in pixels
    9. Average relative length of object, where relative length = current length / mean length
    10. Average width of object in pixels
    11. Average relative width of object
    12. Average aspect ratio of object (width / length)
    13. Average relative aspect ratio (current W/L) / (mean W/L)
    14. Average end wiggle
    body (using whichever end shows a greater angle).
    15. Average number of pixels filled as part of object detection
    """

    # initialize everything
    column_names = ['frame', 'time', 'N', 'N', '5', '6', '7', '8', '9'
                    '10', '11', '12', '13', '14', '15', '%%', 'index']
    frames = []
    times = []
    blob_N = {}
    index_by_blob = {}
    index_by_frame = {}
    lost_and_found = {}
    line_offsets = []
    #frame = 0
    offset = 0
    bID = 0
    bIDs = []
    # organize names and paths
    search_string = test_data_dir + 'blobfiles/' + ex_id + '/*.blobs'
    blobfiles = sorted(glob.glob(search_string))
    basename = blobfiles[0].split('/')[-1].split('_00000k')[0]
    summary_filename = test_data_dir + 'blobfiles/' + ex_id + '/' + basename + '.summary'

    # read all blobs files and pull data
    for i, blobfile in enumerate(blobfiles):
        print 'reading', blobfile.split('/')[-1]
        with open(blobfiles[0], 'r') as f:        
            for line in f:
                line_offsets.append(offset)
                offset += len(line)
                if line[0] == '%':
                    # if id line, make sure index grabs bit offset.
                    bID = int(line[1:])
                    index_by_blob[bID] = [i, line_offsets[-1]]
                elif len(line) > 1:
                    # if data line, make sure frame, 
                    cols = line.split()
                    frame, t = cols[:2]
                    if frame not in frames:
                        frames.append(frame)
                        blob_N[frame] = 0
                        lost_and_found[frame] = []
                        index_by_frame[frame] = []
                    if t not in times:
                        times.append(t)
                    if bID not in bIDs:
                        bIDs.append(bID)
                        # blob is found. 
                        lost_and_found[frame].append('0 {bid}'.format(bid=bID))
                    blob_N[frame] += 1
                elif bID != 0:
                    # blob is lost.
                    lost_and_found[frame].append('{id} 0'.format(id=bID))
                    fID, off = index_by_blob[bID]
                    s = '{bid} {fID}.{off}'.format(bid=bID, fID=fID, off=off)
                    index_by_frame[frame].append(s)

            # when file ends, store remaining bID
            lost_and_found[frame].append('{id} 0'.format(id=bID))
            fID, off = index_by_blob[bID]
            s = '{bid} {fID}.{off}'.format(bid=bID, fID=fID, off=off)
            index_by_frame[frame].append(s)

    #for frame in sorted(lost_and_found):
    #    if len(lost_and_found[frame]):
    #        print frame, lost_and_found[frame]
                    
    assert len(frames) == len(times), 'frames and times have different lengths'
    # TEST
    # this checks to see if the bit offests and files are correct for each blobID
    for bID, (fid, offset) in index_by_blob.iteritems():
        with open(blobfiles[fid], 'r') as f:
            f.seek(offset)
            for line in f:
                assert line == '% {i}\n'.format(i=bID)
                break

    with open(summary_filename, 'w') as f:
        for frame, t in izip(frames, times):
            cols = ['0' for _ in range(15)]
            cols[0] = str(frame)
            cols[1] = str(t)
            cols[2] = cols[3] = str(blob_N[frame])
            # lost and found columns
            l_and_f = lost_and_found[frame]            
            if len(l_and_f) > 0:
                l_and_f = ' '.join(l_and_f)
                cols.append('%%')
                cols.append(l_and_f)
            # blob index columns
            b_index = index_by_frame[frame]            
            if len(b_index) > 0:
                cols.append('%%%')
                cols.append(' '.join(b_index))
            # write columns to summary file.
            f.write(' '.join(cols) + '\n')
    print 'wrote', summary_filename

def write_index_file(ex_id='00000000_000001'):
    """
    """
    index_file_name = '{dir}/0000-00.tsv'.format(dir=LOGISTICS['annotation'])
    print index_file_name
    summary_search = test_data_dir + 'blobfiles/' + ex_id + '/*.summary'    
    sum_file = glob.glob(summary_search)
    if len(sum_file) != 1:
        print 'Warning: no summary file found', summary_search
    sum_file = sum_file[0]
    with open(sum_file, 'r') as f:
        last_time = f.readlines()[-1].split()[1]
    cols = SPREADSHEET['columns']
    data = {}
    for c in cols:
        data[c] = 'test'
    data['ex-id'] = str(ex_id)
    data['vid-flags'] = ''
    data['name'] = sum_file.split('/')[-1].split('.summary')[0]
    data['vid-duration'] = str(last_time)
    data['num-blobs-files'] = '1'    
    data['num-images'] = '0'
    data['vid-flags'] = ''        
    data['pixels-per-mm'] = '45'

    with open(index_file_name, 'w') as f:
        line = '\t'.join(cols)
        f.write(line + '\n')
        line = '\t'.join([data[c] for c in cols])
        f.write(line + '\n')
        
        
if __name__ == '__main__':
    #write_blob_file_for_test_folder()
    #write_summary_file()
    write_index_file()

