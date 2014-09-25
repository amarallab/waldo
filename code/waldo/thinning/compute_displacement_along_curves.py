"""
    given two curves at two consecutive times,
    the module computes how much the later curve
    is displaced along the older.
    to do this, we project the displacemnet of very point
    along the line along the direction of the curve
    (consecutive points along the curve)
    displacement is positive is the worm is moving
    in the direction of the curve
"""
from __future__ import absolute_import

import math
#from pylab import *

def get_direction_and_distance(p1, p2, m1, m2, perpendicular=False):
    """
        p1, p2, m1, and m2 are all tuples (x,y)
        the function computes the projection of the
        vector m2-m1 along the vector p2-p1
        if the projection is positive it means
        that m2 is getting closer to p2
        the number tells how much it got closer
        N.B. the perpendicular_projection is always
        positive: we want to know how much is moving
        perpendicularly to the curve
        but without caring if it is left ot right
    """

    vector_p2_p1 = (p2[0] - p1[0], p2[1] - p1[1])
    vector_m2_m1 = (m2[0] - m1[0], m2[1] - m1[1])

    norm_p2_p1 = math.sqrt(vector_p2_p1[0] ** 2 + vector_p2_p1[1] ** 2)
    norm_m2_m1 = math.sqrt(vector_m2_m1[0] ** 2 + vector_m2_m1[1] ** 2)
    if norm_p2_p1 == 0 or norm_m2_m1 == 0:
        return 0.

    dot_product = vector_p2_p1[0] * vector_m2_m1[0] + \
                  vector_p2_p1[1] * vector_m2_m1[1]

    if perpendicular:
    #use sin
        norm_m2_m1_square = vector_m2_m1[0] ** 2 + vector_m2_m1[1] ** 2
        cosine_ = dot_product / (norm_p2_p1) / math.sqrt(norm_m2_m1_square)
        # to guard against round off errors
        if (1. - cosine_ ** 2) < 0:
            sin_ = 0.
        else:
            sin_ = math.sqrt(1. - cosine_ ** 2)
        perpendicular_projection = math.fabs(math.sqrt(norm_m2_m1_square) * sin_)

        # this can be skipped setting check_=False
        check_ = False
        if check_:
            # check with pitagora's theorem
            projection = dot_product / norm_p2_p1
            perpendicular_projection2 = math.sqrt(norm_m2_m1_square - projection ** 2)
            assert math.fabs(perpendicular_projection - perpendicular_projection2) < 1e-8
            # check with line intersections
            if vector_p2_p1[0] == 0:
                perpendicular_projection3 = math.fabs(vector_m2_m1[0])
            else:
                ratio = vector_p2_p1[1] / vector_p2_p1[0]
                a = 1. / math.sqrt(1 + ratio ** 2)
                perpendicular_projection3 = vector_m2_m1[0] * (-ratio) * a + \
                                            vector_m2_m1[1] * a
                perpendicular_projection3 = math.fabs(perpendicular_projection3)
            assert math.fabs(perpendicular_projection - perpendicular_projection3) < 1e-8

        return perpendicular_projection
    else:
        return -1 * dot_product / norm_p2_p1


def compute_displacement_along_curve(curve1, curve2, perpendicular=False, points='all'):
    """
        curve1 and curve2 are lists of tuples (x,y)
        the function return the sum of the
        projections of (curve2-curve1) along curve1
        if perpendicular, the displacement is computed
        perpendicular to the spine
    """

    if points == 'all': points = range(len(curve1))

    total_displacement = 0.
    for i in points:
        if i + 1 < len(curve1):
            p2 = curve1[i + 1]
            p1 = curve1[i]
            p1_later = curve2[i]
            # positive number if p1_later gets closer to p2
            dist = get_direction_and_distance(p1, p2,
                                              p1, p1_later,
                                              perpendicular=perpendicular)
            total_displacement += dist

    return total_displacement












