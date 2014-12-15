# __author__ = 'heltena'
#
# import os
#
# from PyQt4 import QtGui
#
# import numpy as np
#
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
#
#
# def perp(v):
#     # adapted from http://stackoverflow.com/a/3252222/194586
#     p = np.empty_like(v)
#     p[0] = -v[1]
#     p[1] = v[0]
#     return p
#
#
# def circle_3pt(a, b, c):
#     """
#     1. Make some arbitrary vectors along the perpendicular bisectors between
#         two pairs of points.
#     2. Find where they intersect (the center).
#     3. Find the distance between center and any one of the points (the
#         radius).
#     """
#
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)
#
#     # find perpendicular bisectors
#     ab = b - a
#     c_ab = (a + b) / 2
#     pb_ab = perp(ab)
#     bc = c - b
#     c_bc = (b + c) / 2
#     pb_bc = perp(bc)
#
#     ab2 = c_ab + pb_ab
#     bc2 = c_bc + pb_bc
#
#     # find where some example vectors intersect
#     #center = seg_intersect(c_ab, c_ab + pb_ab, c_bc, c_bc + pb_bc)
#
#     A1 = ab2[1] - c_ab[1]
#     B1 = c_ab[0] - ab2[0]
#     C1 = A1 * c_ab[0] + B1 * c_ab[1]
#     A2 = bc2[1] - c_bc[1]
#     B2 = c_bc[0] - bc2[0]
#     C2 = A2 * c_bc[0] + B2 * c_bc[1]
#     center = np.linalg.inv(np.matrix([[A1, B1],[A2, B2]])) * np.matrix([[C1], [C2]])
#     center = np.array(center).flatten()
#     radius = np.linalg.norm(a - center)
#     return center, radius
#
#
# class CacheThresholdDataPage(QtGui.QWizardPage):
#     def __init__(self, data, parent=None):
#         super(CacheThresholdDataPage, self).__init__(parent)
#         self.data = data
#         self.setTitle("Cache Threshold Data")
#
#         label = QtGui.QLabel("...")
#
#         self.histogram_figure = plt.figure()
#         self.histogram_canvas = FigureCanvas(self.histogram_figure)
#
#         self.image_figure = plt.figure()
#         self.image_canvas = FigureCanvas(self.image_figure)
#         self.toolbar = NavigationToolbar(self.image_canvas, self)
#
#         # First row
#         first_row_layout = QtGui.QHBoxLayout()
#         first_row_layout.addWidget(self.histogram_canvas)
#         first_row_layout.addWidget(self.image_canvas)
#
#         self.circle = None
#         self.circle_pos = (0, 0)
#         self.circle_radius = 1
#
#         layout = QtGui.QVBoxLayout()
#         layout.addWidget(label)
#         layout.addLayout(first_row_layout)
#         layout.addWidget(self.toolbar)
#         self.setLayout(layout)
#
#         # times, impaths = grab_images_in_time_range(self.current_id, start_time=0)
#         # times = [float(t) for t in times]
#         # times, impaths = zip(*sorted(zip(times, impaths)))
#         #
#         # self.background = CacheThresholdDataPage.create_background(impaths)
#
#     @staticmethod
#     def create_background(impaths):
#         """
#         create a background image for background subtraction.
#         The background image is the maximum pixel values from three grayscale images.
#
#         params
#         ---------
#         impaths: (list)
#            this is a sorted list containing paths to all the image files from one recording.
#         """
#         first = mpimg.imread(impaths[0])
#         mid = mpimg.imread(impaths[int(len(impaths)/2)])
#         last = mpimg.imread(impaths[-1])
#         return np.maximum(np.maximum(first, mid), last)
