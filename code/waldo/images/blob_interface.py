import functools
import numpy as np
import six

from waldo.extern import multiworm
from multiworm.readers import blob as blob_reader

__author__ = 'peterwinter'


def find_nearest_index(seq, value):
    return (np.abs(np.array(seq)-value)).argmin()


def frame_parser(blob_lines, frame):
    """
    A lighter, probably quicker, parser to just get a single frame of
    data out of a blob.
    """
    first_line = six.next(blob_lines)
    frame_offset = frame - int(first_line.split(' ', 1)[0])
    line = first_line

    # blindly consume as many lines as needed
    try:
        for dummy in range(frame_offset):
            line = six.next(blob_lines)
    except multiworm.core.MWTDataError:
        pass

    # parse the line and return
    blob = blob_reader.parse([line])
    if blob['frame'][0] != frame:
        raise multiworm.core.MWTDataError("Blob line offset failure")
    return blob


def frame_parser_spec(frame):
    return functools.partial(frame_parser, frame=frame)


def grab_blob_data(experiment, time):
    """
    pull the frame number and a list of tuples (bid, centroid, outline)
    for a given time and experiment.

    params
    -------
    experiment: (experiment object from wio)
        cooresponds to a specific ex_id
    time: (float)
        the closest time in seconds for which we would like to retrieve data

    returns
    -------
    frame: (int)
        the frame number that most closely matches the given time.
    blob_data: (list of tuples)
        the list contains the (blob_id [str], centroid [xy tuple], outlines [list of points])
        for all blobs tracked during that particular frame.
    """

    # get the objects from MWT blobs files.
    frame = find_nearest_index(experiment.frame_times, time) + 1
    bids = experiment.blobs_in_frame(frame)
    #outlines, centroids, outline_ids = [], [], []
    parser = frame_parser_spec(frame)
    blob_data = []
    bad_blobs = []

    for bid in bids:
        try:
            blob = experiment._parse_blob(bid, parser)
            if blob['contour_encode_len'][0]:
                outline = blob_reader.decode_outline(
                    blob['contour_start'][0],
                    blob['contour_encode_len'][0],
                    blob['contour_encoded'][0],
                )
                blob_data.append((bid, blob['centroid'][0], outline))
        except ValueError:
            bad_blobs.append(bid)
    if bad_blobs:
        if len(bad_blobs)> 5:
            print('Warning: {n} blobs failed to load data'.format(n=len(bad_blobs)))
        else:
            print('Warning: {n} blobs failed to load data {ids}'.format(n=len(bad_blobs), ids=bad_blobs))
    return frame, blob_data