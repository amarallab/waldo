__author__ = 'heltena'

import os

from PyQt4 import QtGui
from PyQt4.QtGui import QSizePolicy
from PyQt4.QtCore import Qt

import numpy as np
from scipy import ndimage
import glob
import time
import json
import errno

import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import matplotlib.image as mpimg
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from skimage import morphology
from skimage.measure import regionprops
import tasking

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


class CacheThresholdLoadingDialog(QtGui.QDialog):
    def __init__(self, ex_id, func, finish_func, parent=None):
        super(CacheThresholdLoadingDialog, self).__init__(parent)
        self.finish_func = finish_func

        label = QtGui.QLabel("Loading experiment: {ex_id}".format(ex_id=ex_id))
        progress_bar = QtGui.QProgressBar()
        progress_bar.setRange(0, 100)

        cancel_run_button = QtGui.QPushButton("Cancel")
        cancel_run_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        cancel_run_button.clicked.connect(self.cancel_run_button_clicked)

        progress_layout = QtGui.QHBoxLayout()
        progress_layout.addWidget(progress_bar)
        progress_layout.addWidget(cancel_run_button)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(label)
        layout.addLayout(progress_layout)
        self.setLayout(layout)

        self.progress_bar = progress_bar
        self.cancel_run_button = cancel_run_button

        self.task = tasking.CommandTask(self.madeProgress, self.finished, self.cancelled)
        self.task.start(func)
        self.setFixedSize(self.minimumSize())
        self.setWindowFlags(Qt.Tool | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.CustomizeWindowHint)


    def cancel_run_button_clicked(self):
        self.cancel_run_button.setEnabled(False)
        if self.task is not None:
            self.task.requestCancel()
        return False

    def madeProgress(self, item, value):
        if self.task is not None:
            self.progress_bar.setValue(value * 100)

    def finished(self):
        self.task.waitFinished()
        self.task = None
        self.result = "Finished"
        self.close()
        self.finish_func()

    def cancelled(self):
        self.task.waitFinished()
        self.task = None
        self.result = "Cancelled"
        self.close()

    def closeEvent(self, ev):
        if self.task is None:
            ev.accept()
        else:
            ev.ignore()


class CacheThresholdDataPage(QtGui.QWizardPage):
    def __init__(self, data, parent=None):
        super(CacheThresholdDataPage, self).__init__(parent)
        self.data = data
        self.setTitle("Cache Threshold Data")

        label = QtGui.QLabel("...")

        self.histogram_figure = plt.figure()
        self.histogram_canvas = FigureCanvas(self.histogram_figure)

        self.image_figure = plt.figure()
        self.image_canvas = FigureCanvas(self.image_figure)
        self.toolbar = NavigationToolbar(self.histogram_canvas, self)
        self.toolbar.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        gs = grd.GridSpec(2, 1)
        self.ax_objects = self.histogram_figure.add_subplot(gs[0, 0])
        self.ax_area = self.histogram_figure.add_subplot(gs[1, 0], sharex=self.ax_objects)
        self.ax_image = self.image_figure.add_subplot(111)

        # First row
        first_row_layout = QtGui.QHBoxLayout()
        first_row_layout.addWidget(self.histogram_canvas)
        first_row_layout.addWidget(self.image_canvas)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(label)
        layout.addLayout(first_row_layout)
        layout.addWidget(self.toolbar)
        self.setLayout(layout)

        self.label = label
        self.thresholds = []

        self.histogram_figure.canvas.mpl_connect('button_press_event', self.on_histogram_button_pressed)
        self.image_figure.canvas.mpl_connect('button_press_event', self.on_image_button_pressed)

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

    def initializePage(self):
        ex_id = self.data.experiment_id
        self.label.setText("Experiment: {ex_id}".format(ex_id=ex_id))
        self.current_threshold = 0.0005
        data = {}
        if ex_id is not None:
            self.annotation_filename = os.path.join(self.data.waldo_folder, 'waldo_data', ex_id) + '-thresholddata.json'
            try:
                with open(self.annotation_filename, "rt") as f:
                    data = json.loads(f.read())
            except IOError as ex:
                pass

        self.circle = None
        self.circle_pos = (data.get('y', 0), data.get('x', 0))  # stored transposed!!
        self.circle_radius = data.get('r', 1)
        self.current_threshold = data.get('threshold', 0.0005)

        self.data.threshold = self.current_threshold
        self.data.circle_pos = self.circle_pos
        self.data.circle_radius = self.circle_radius

        times, impaths = self.grab_images_in_time_range(self.data.experiment_id)
        if len(times) > 0:
            times = [float(t) for t in times]
            times, impaths = zip(*sorted(zip(times, impaths)))

        if len(impaths) == 0:
            self.background = None
            self.mid_image = None
        else:
            self.background = CacheThresholdDataPage.create_background(impaths)
            self.mid_image = mpimg.imread(impaths[int(len(impaths)/2)])
        self.mouse_points = []

        dlg = CacheThresholdLoadingDialog(ex_id, self.calculate_threshold, self.finished, self)
        dlg.setModal(True)
        dlg.exec_()

    @staticmethod
    def mkdir_p(path):
        try:
            os.makedirs(path)
        except OSError as exc: # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else: raise

    def save_data(self):
        # note: the image is usually transposed. we didn't here,
        # so x and y are flipped during saving process.
        data = {'threshold': self.current_threshold,
                'x': self.circle_pos[1],
                'y': self.circle_pos[0],
                'r': self.circle_radius}

        CacheThresholdDataPage.mkdir_p(os.path.dirname(self.annotation_filename))
        with open(self.annotation_filename, "wt") as f:
            f.write(json.dumps(data, indent=4))

        self.data.threshold = self.current_threshold
        self.data.circle_pos = self.circle_pos
        self.data.circle_radius = self.circle_radius

    def calculate_threshold(self, callback):
        self.thresholds = []
        for i, t in enumerate(np.linspace(start=0.00001, stop=0.001, num=30)):
            valid, N, m, s = self.data_from_threshold(t)
            if valid:
                self.thresholds.append((t, N, m, s))
            callback(0, i / 30.)
        callback(0, 1)

    def finished(self):
        if len(self.thresholds) == 0:
            self.ax_objects.clear()
            self.ax_area.clear()
            self.ax_image.clear()
            return

        x, ns, means, stds = zip(*self.thresholds)
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

        self.line_objects = self.ax_objects.plot((self.current_threshold, self.current_threshold), (-10000, 10000), '--', color='red')
        self.line_area = self.ax_area.plot((self.current_threshold, self.current_threshold), (-1000000, 1000000), '--', color='red')
        self.show_threshold()

    def isComplete(self):
        return self.circle_pos[0] != 0 or self.circle_pos[1] != 0

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
        mask = self.create_binary_mask(self.mid_image, self.background, self.current_threshold)
        self.ax_image.clear()
        self.ax_image.imshow(self.mid_image, cmap=plt.cm.gray, interpolation='nearest')
        self.ax_image.contour(mask, [0.5], linewidths=1.2, colors='b')
        self.ax_image.set_title('threshold = {t}'.format(t=self.current_threshold))
        self.ax_image.axis('off')
        self.circle = None
        self.update_image_circle()

    def update_image_circle(self):
        if self.circle is not None:
            self.circle.center = self.circle_pos
            self.circle.radius = self.circle_radius
            self.image_figure.canvas.draw()
        else:
            self.circle = plt.Circle(self.circle_pos, self.circle_radius, color=(1, 0, 0, 0.25))
            self.ax_image.add_artist(self.circle)
        self.image_figure.canvas.draw()

    def create_image_directory(self, ex_id):
        search_path = os.path.join(self.data.waldo_folder, 'raw_data', ex_id) + '/*.png'
        images = glob.glob(search_path)
        if not images:
            print('something may be wrong with search path. no images found:', search_path)
            return {}
        basename = ['z' for _ in range(1000)]
        for i in images:
            if len(i) < len(basename):
                basename = i
        basename = basename.split('.png')[0]

        time_to_image = {}
        for i in images:
            time = i.split(basename)[-1].split('.png')[0]
            if time:
                if time + '.png' == i:
                    print('i', i)
                    print('base', basename)
                    print('time', time)
                    continue
                time = '{t}'.format(t=round(int(time) / 1000.0), ndigits=3)
            else:
                time = '0.000'
            time_to_image[time] = i
        return time_to_image

    def on_histogram_button_pressed(self, ev):
        if self.current_threshold != ev.xdata:
            self.current_threshold = ev.xdata

            self.line_objects[0].remove()
            self.line_area[0].remove()
            self.line_objects = self.ax_objects.plot((self.current_threshold, self.current_threshold), (-10000, 10000), '--', color='red')
            self.line_area = self.ax_area.plot((self.current_threshold, self.current_threshold), (-1000000, 1000000), '--', color='red')

            self.show_threshold()
            self.histogram_figure.canvas.draw()
            self.save_data()

    def on_image_button_pressed(self, ev):
        if ev.button == 3:
            self.mouse_points = []
            self.circle_pos = (ev.xdata, ev.ydata)
            self.update_image_circle()
            self.save_data()
        else:
            self.mouse_points.append((ev.xdata, ev.ydata))
            if len(self.mouse_points) == 3:
                center, radius = circle_3pt(*self.mouse_points)
                self.circle_pos = center
                self.circle_radius = radius
                self.update_image_circle()
                self.mouse_points = []
                self.completeChanged.emit()
                self.save_data()

    def grab_images_in_time_range(self, ex_id):
        time_to_image = self.create_image_directory(ex_id)
        image_times, image_paths = [], []
        for im_time, im_path in time_to_image.iteritems():
            image_times.append(im_time)
            image_paths.append(im_path)
        return image_times, image_paths
