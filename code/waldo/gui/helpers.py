# standard library
import json

# third party
import numpy as np

# project specific
from waldo.wio import paths

def experiment_has_thresholdCache(experiment_id):
    if experiment_id is None:
        return False

    data = {}
    annotation_filename = paths.threshold_data(experiment_id)
    try:
        with open(str(annotation_filename), "rt") as f:
            data = json.loads(f.read())
    except IOError as ex:
        pass

    return 'threshold' in data and 'r' in data and 'x' in data and 'y' in data


def perp(v):
    # adapted from http://stackoverflow.com/a/3252222/194586
    p = np.empty_like(v)
    p[0] = -v[1]
    p[1] = v[0]
    return p


def circle_3pt(a, b, c):
    """
    1. Make some arbitrary vectors along the perpendicular bisectors between
        two pairs of points.
    2. Find where they intersect (the center).
    3. Find the distance between center and any one of the points (the
        radius).
    """

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # find perpendicular bisectors
    ab = b - a
    c_ab = (a + b) / 2
    pb_ab = perp(ab)
    bc = c - b
    c_bc = (b + c) / 2
    pb_bc = perp(bc)

    ab2 = c_ab + pb_ab
    bc2 = c_bc + pb_bc

    # find where some example vectors intersect
    #center = seg_intersect(c_ab, c_ab + pb_ab, c_bc, c_bc + pb_bc)

    A1 = ab2[1] - c_ab[1]
    B1 = c_ab[0] - ab2[0]
    C1 = A1 * c_ab[0] + B1 * c_ab[1]
    A2 = bc2[1] - c_bc[1]
    B2 = c_bc[0] - bc2[0]
    C2 = A2 * c_bc[0] + B2 * c_bc[1]
    center = np.linalg.inv(np.matrix([[A1, B1],[A2, B2]])) * np.matrix([[C1], [C2]])
    center = np.array(center).flatten()
    radius = np.linalg.norm(a - center)
    return center, radius
