from PyQt4 import QtGui
from PyQt4.QtGui import QSizePolicy
from PyQt4.QtCore import Qt, QTimer

import os

import numpy as np
from scipy import ndimage
import json
import errno
from waldo.wio import Experiment

import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import matplotlib.image as mpimg
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from skimage import morphology
from skimage.measure import regionprops

# from waldo.images.grab_images import grab_images_in_time_range
from waldo.gui import tasking
from waldo.wio import paths
from .loaders import CacheThresholdLoadingDialog


class ThresholdCacheWidget(QtGui.QGridLayout):
    def __init__(self, on_changed_ev, parent=None):
        super(ThresholdCacheWidget, self).__init__()
        self.on_changed_ev = on_changed_ev
        self.parent = parent

        self.histogram_figure = plt.figure()
        self.histogram_canvas = FigureCanvas(self.histogram_figure)
        self.histogram_canvas.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.histogram_canvas.setMinimumSize(50, 50)
        self.histogram_toolbar = NavigationToolbar(self.histogram_canvas, parent)
        self.histogram_toolbar.coordinates = False
        self.histogram_toolbar.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        self.image_figure = plt.figure()
        self.image_canvas = FigureCanvas(self.image_figure)
        self.image_canvas.setMinimumSize(50, 50)
        self.image_canvas.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.image_toolbar = NavigationToolbar(self.image_canvas, parent)
        self.image_toolbar.coordinates = False
        self.image_toolbar.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        gs = grd.GridSpec(2, 1)
        self.ax_objects = self.histogram_figure.add_subplot(gs[0, 0])
        self.ax_area = self.histogram_figure.add_subplot(gs[1, 0], sharex=self.ax_objects)
        self.ax_image = self.image_figure.add_subplot(111)

        q1 = QtGui.QLabel("<b>Choose Threshold</b>")
        # q1.setAlignment(Qt.AlignHCenter)
        self.addWidget(q1, 0, 0, 1, 1)
        self.addWidget(QtGui.QLabel("Click on either graph to pick a threshold value"), 1, 0, 1, 1)
        self.addWidget(self.histogram_canvas, 2, 0, 1, 1)
        self.addWidget(self.histogram_toolbar, 3, 0, 1, 1)

        q2 = QtGui.QLabel("<b>Define Region of Interest</b>")
        # q2.setAlignment(Qt.AlignHCenter)
        self.addWidget(q2, 0, 1, 1, 1)
        self.addWidget(QtGui.QLabel("Click on image three times to define the region of interest"), 1, 1, 1, 1)
        self.addWidget(self.image_canvas, 2, 1, 1, 1)
        self.addWidget(self.image_toolbar, 3, 1, 1, 1)

        self.histogram_figure.canvas.mpl_connect('button_press_event', self.on_histogram_button_pressed)
        self.image_figure.canvas.mpl_connect('button_press_event', self.on_image_button_pressed)

        self.roi_center = [0, 0]
        self.roi_radius = 1
        self.thresholds = []

    @staticmethod
    def create_background(impaths):
        """
        create a background image for background subtraction.
        The background image is the maximum pixel values from three grayscale images.

        params
        ---------
        impaths: (list)
           this is a sorted list containing paths to all the image files from one recording.
        """
        if len(impaths) == 0:
            return None
        first = mpimg.imread(impaths[0])
        mid = mpimg.imread(impaths[int(len(impaths)/2)])
        last = mpimg.imread(impaths[-1])
        return np.maximum(np.maximum(first, mid), last)

    def clear_experiment_data(self):
        self.circle = None
        self.roi_center = (0, 0)
        self.roi_radius = 1
        self.threshold = 0.0005

    def load_experiment(self, experiment):
        self.experiment = experiment
        self.annotation_filename = str(paths.threshold_data(experiment.id))
        try:
            with open(self.annotation_filename, "rt") as f:
                data = json.loads(f.read())
            self.circle = None
            self.roi_center = (data.get('y', 0), data.get('x', 0))  # stored transposed!!
            self.roi_radius = data.get('r', 1)
            self.threshold = data.get('threshold', 0.0005)
        except IOError as ex:
            self.clear_experiment_data()

        times, impaths = zip(*sorted(experiment.image_files.items()))
        impaths = [str(s) for s in impaths]

        if times is not None and len(times) > 0:
            times = [float(t) for t in times]
            times, impaths = zip(*sorted(zip(times, impaths)))

        if impaths is None or len(impaths) == 0:
            self.background = None
            self.mid_image = None
        else:
            self.background = ThresholdCacheWidget.create_background(impaths)
            self.mid_image = mpimg.imread(impaths[int(len(impaths)/2)])
        self.mouse_points = []
        QTimer.singleShot(0, self.show_dialog)

    def show_dialog(self):
        dlg = CacheThresholdLoadingDialog(self.experiment.id, self.calculate_threshold, self.finished, self.parent)
        dlg.setModal(True)
        dlg.exec_()

    def calculate_threshold(self, callback):
        self.thresholds = []
        for i, t in enumerate(np.linspace(start=0.00001, stop=0.001, num=30)):
            valid, N, m, s = self.data_from_threshold(t)
            if valid:
                self.thresholds.append((t, N, m, s))
            callback(0, i / 30.)
        callback(0, 1)

    def finished(self):
        self.update_data(self.thresholds, self.threshold)

    def isComplete(self):
        return self.roi_center[0] != 0 or self.roi_center[1] != 0

    def create_binary_mask(self, img, background, threshold, minsize=100):
        """
        creates a binary array the same size as the image with 1s denoting objects
        and 0s denoting background.

        params
        --------
        img: (image ie. numpy array)
            each pixel denotes greyscale pixel intensities.
        background: (image ie. numpy array)
            the background image with maximum pixel intensities (made with create_background)
        threshold: (float)
            the threshold value used to create the binary mask after pixel intensities for (background - image) have been calculated.
        minsize: (int)
            the fewest allowable pixels for an object. objects with an area containing fewer pixels are removed.
        """
        if img is None or background is None:
            return None
        mask = (background - img) > threshold
        result = morphology.remove_small_objects(mask, minsize)
        return result

    def data_from_threshold(self, threshold):
        if self.mid_image is None:
            return False, None, None, None
        mask = self.create_binary_mask(self.mid_image, self.background, threshold=threshold)
        labels, N = ndimage.label(mask)
        sizes = [r.area for r in regionprops(labels)]
        if len(sizes) == 0:
            return False, None, None, None
        else:
            m, s = np.mean(sizes), np.std(sizes)
            return True, N, m, s

    @staticmethod
    def mkdir_p(path):
        try:
            os.makedirs(path)
        except OSError as exc: # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else: raise

    def save_data(self):
        if self.annotation_filename is None:
            return

        # note: the image is usually transposed. we didn't here,
        # so x and y are flipped during saving process.
        data = {'threshold': self.threshold,
                'x': self.roi_center[1],
                'y': self.roi_center[0],
                'r': self.roi_radius}

        ThresholdCacheWidget.mkdir_p(os.path.dirname(self.annotation_filename))
        with open(self.annotation_filename, "wt") as f:
            f.write(json.dumps(data, indent=4))

    def update_data(self, thresholds, current_threshold):
        if len(thresholds) == 0:
            self.ax_objects.clear()
            self.ax_area.clear()
            self.ax_image.clear()
            self.line_objects = None
            self.line_area = None
            return

        x, ns, means, stds = zip(*thresholds)
        final_t = x[-1]

        # make the plot
        self.ax_objects.clear()
        self.ax_objects.plot(x, ns, '.--')
        self.ax_objects.set_ylabel('N objects')
        self.ax_objects.set_ylim([0, 150])

        top = np.array(means) + np.array(stds)
        bottom = np.array(means) - np.array(stds)

        self.ax_area.clear()
        self.ax_area.plot(x, means, '.--', color='blue')
        self.ax_area.plot(x, top, '--', color='green')
        self.ax_area.plot(x, bottom, '--', color='green')
        self.ax_area.axvline(x=.5, ymin=0, ymax=1)

        self.ax_area.set_ylim([0, 600])
        self.ax_area.set_ylabel('mean area')
        self.ax_objects.set_xlim([0, final_t])

        self.line_objects = self.ax_objects.plot((current_threshold, current_threshold), (-10000, 10000), '--', color='red')
        self.line_area = self.ax_area.plot((current_threshold, current_threshold), (-1000000, 1000000), '--', color='red')
        self.show_threshold()

    def show_threshold(self):
        """
        plots an image with the outlines of all objects overlaid on top.

        params
        --------
        img: (image ie. numpy array)
            each pixel denotes greyscale pixel intensities.
        background: (image ie. numpy array)
            the background image with maximum pixel intensities (made with create_background)
        threshold: (float)
            the threshold value used to create the binary mask after pixel intensities for (background - image) have been calculated.
        """
        mask = self.create_binary_mask(self.mid_image, self.background, self.threshold)
        self.ax_image.clear()
        self.ax_image.imshow(self.mid_image, cmap=plt.cm.gray, interpolation='nearest')
        self.ax_image.contour(mask, [0.5], linewidths=1.2, colors='b')
        self.ax_image.axis('off')
        self.circle = None
        self.update_image_circle()

    def update_image_circle(self):
        if self.circle is not None:
            self.circle.center = self.roi_center
            self.circle.radius = self.roi_radius
            self.image_figure.canvas.draw()
        else:
            self.circle = plt.Circle(self.roi_center, self.roi_radius, color=(1, 0, 0, 0.25))
            self.ax_image.add_artist(self.circle)
        self.image_figure.canvas.draw()

    def on_histogram_button_pressed(self, ev):
        if self.threshold != ev.xdata:
            self.threshold = ev.xdata

            if self.line_objects is not None and len(self.line_objects) > 0:
                self.line_objects[0].remove()
            if self.line_area is not None and len(self.line_area) > 0:
                self.line_area[0].remove()
            self.line_objects = self.ax_objects.plot((self.threshold, self.threshold), (-10000, 10000), '--', color='red')
            self.line_area = self.ax_area.plot((self.threshold, self.threshold), (-1000000, 1000000), '--', color='red')

            self.show_threshold()
            self.histogram_figure.canvas.draw()
            self.save_data()

    def on_image_button_pressed(self, ev):
        if ev.button == 3:
            self.mouse_points = []
            self.roi_center = (ev.xdata, ev.ydata)
            self.update_image_circle()
            self.save_data()
        else:
            self.mouse_points.append((ev.xdata, ev.ydata))
            if len(self.mouse_points) == 3:
                center, radius = circle_3pt(*self.mouse_points)
                self.roi_center = center
                self.roi_radius = radius
                self.update_image_circle()
                self.mouse_points = []
                self.save_data()
                self.on_changed_ev()

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
