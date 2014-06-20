# This notebook is for finding the segmentation threshold that most clearly finds worms in a recording.
# It is intended as an alternative method of validating the MultiWorm Tracker's results.

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
from matplotlib.widgets import Button, Slider
import json

import matplotlib.image as mpimg
import matplotlib.cm as cm
from scipy import ndimage
from skimage import morphology
from skimage.measure import regionprops

# # Path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
SHARED_DIR = os.path.abspath(os.path.join(HERE, '..'))
PROJECT_DIR = os.path.abspath(os.path.join(SHARED_DIR, '..'))
print HERE
print SHARED_DIR
print PROJECT_DIR

sys.path.append(SHARED_DIR)
sys.path.append(PROJECT_DIR)

# nonstandard imports
from grab_images import grab_images_in_time_range
from settings.local import LOGISTICS
from wio.file_manager import ensure_dir_exists

MWT_DIR = LOGISTICS['filesystem_data']
DATA_DIR = LOGISTICS['filesystem_data']
PRETREATMENT_DIR = os.path.abspath(LOGISTICS['pretreatment'])
ensure_dir_exists(PRETREATMENT_DIR)


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

INCR_X = 1
INCR_Y = 1
INCR_RADIUS = 1

class InteractivePlot:
    def __init__(self, ids, annotation_filename, cache_dir, initial_threshold=0.0005):
        self.ids = ids
        self.current_index = 0
        self.current_id = None
        self.annotation_filename = annotation_filename
        self.cache_dir = cache_dir
        self.current_threshold = initial_threshold
        self.thresholds = []

        self.data = None
        self.load_data()

        self.fig = plt.figure()
        self.fig.canvas.mpl_connect('button_press_event', self.on_button_pressed)
        self.fig.canvas.mpl_connect('button_release_event', self.on_button_released)

        gs = grd.GridSpec(14, 3)
        gs.update(left=0.05, right=0.95, wspace=0.05, hspace=0.15)
        self.ax_objects = self.fig.add_subplot(gs[0:7, 0])
        self.ax_area = self.fig.add_subplot(gs[7:13, 0], sharex=self.ax_objects)
        self.ax_image = self.fig.add_subplot(gs[0:13, 1:3])

        ax_title = plt.axes([0, 0.95, 1, 0.05])
        self.title = Button(ax_title, 'Title')

        ax_prev = plt.axes([0, 0, 0.1, 0.05])
        self.prev = Button(ax_prev, 'Prev')
        self.prev.on_clicked(self.on_prev_clicked)

        ax_next = plt.axes([0.1, 0, 0.1, 0.05])
        self.next = Button(ax_next, 'Next')
        self.next.on_clicked(self.on_next_clicked)

        ax_save = plt.axes([0.3, 0, 0.1, 0.05])
        self.save = Button(ax_save, 'Save')
        self.save.on_clicked(self.on_save_clicked)

        ax_centerx = plt.axes([0.5, 0, 0.1, 0.05])
        self.centerx = Button(ax_centerx, '< X >')
        self.centerx.on_clicked(self.on_centerx_clicked)

        ax_centery = plt.axes([0.65, 0, 0.1, 0.05])
        self.centery = Button(ax_centery, '< Y >')
        self.centery.on_clicked(self.on_centery_clicked)

        ax_radius = plt.axes([0.8, 0, 0.1, 0.05])
        self.radius = Button(ax_radius, '< R >')
        self.radius.on_clicked(self.on_radius_clicked)

        self.circle = None
        self.circle_pos = (0, 0)
        self.circle_radius = 1

        # list of thresholds to try out
        self.load_index(0)

    def load_index(self, index):
        self.current_index = index
        self.current_id = self.ids[index]

        self.title.label.set_text('Current: %s' % self.current_id)
        self.fig.canvas.draw()

        # grab images and times.
        times, impaths = grab_images_in_time_range(self.current_id, start_time=0)
        times = [float(t) for t in times]
        times, impaths = zip(*sorted(zip(times, impaths)))

        self.background = InteractivePlot.create_background(impaths)

        if self.current_id in self.data:
            d = self.data[self.current_id]
            self.current_threshold = d['threshold']
            self.circle_pos = (d['y'], d['x']) #note xy are purposely switched
            self.circle_radius = d['r']
        else:
            self.circle_pos = (0, 0)
            self.circle_radius = max(self.background.shape)/2

        self.centerx.label.set_text("< X: %d >" % self.circle_pos[0])
        self.centery.label.set_text("< Y: %d >" % self.circle_pos[1])
        self.radius.label.set_text("< R: %d >" % self.circle_radius)

        # pick an image to test. the middle one is good.
        self.mid_image = mpimg.imread(impaths[int(len(impaths)/2)])

        self.mouse_points = [] # for generate circle from 3 points
        # run functions.
        thresholds = np.linspace(start=0.00001, stop=0.001, num=30)
        self.show_threshold_properties(thresholds)
        #NO UNCOMMENT show_threshold_spread(mid, background)

        self.show_threshold()

    def load_data(self):
        try:
            with open(self.annotation_filename, "rt") as f:
                self.data = json.loads(f.read())
        except IOError, ex:
            self.data = {}
            print "E: %s (%s)" % (os.strerror(ex.errno), self.annotation_filename)

    def save_data(self):
        # note: the image is usually transposed. we didn't here,
        # so x and y are flipped during saving process.
        self.data[self.current_id] = {'threshold': self.current_threshold,
                                      'x': self.circle_pos[1],
                                      'y': self.circle_pos[0],
                                      'r': self.circle_radius}
        with open(self.annotation_filename, "wt") as f:
            f.write(json.dumps(self.data, indent=4))

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
        first = mpimg.imread(impaths[0])
        mid = mpimg.imread(impaths[int(len(impaths)/2)])
        last = mpimg.imread(impaths[-1])
        return np.maximum(np.maximum(first, mid), last)

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
        mask = (background - img) > threshold
        result = morphology.remove_small_objects(mask, minsize)
        return result

    def data_from_threshold(self, threshold):
        mask = self.create_binary_mask(self.mid_image, self.background, threshold=threshold)
        labels, N = ndimage.label(mask)
        sizes = [r.area for r in regionprops(labels)]
        if len(sizes) == 0:
            return False, None, None, None
        else:
            m, s = np.mean(sizes), np.std(sizes)
            return True, N, m, s

    def show_threshold_properties(self, thresholds):
        """
        plots the number, mean area, and std area of objects found for each threshold value specified.

        params
        --------
        img: (image ie. numpy array)
            each pixel denotes greyscale pixel intensities.
        ax0: axes on N objects will be drawn
        ax1: axes on mean data will be drawn
        background: (image ie. numpy array)
            the background image with maximum pixel intensities (made with create_background)
        thresholds: (list of floats)
            a list of threshold values to calculate. should be sorted from least to greatest.
        """
        cache_thresholds = {}
        filename = os.path.join(self.cache_dir,
                                "cache-{cid}.json".format(cid=self.current_id))
        try:
            with open(filename, "r") as f:
                x = json.load(f)
            for k, v in x.iteritems():
                cache_thresholds[float(k)] = v
        except:
            print "NO cache threshold for %s" % self.current_id

        self.thresholds = []
        for i, t in enumerate(thresholds):
            if t in cache_thresholds:
                x = cache_thresholds[t]
                valid = x['valid']
                N = x['N']
                m = x['m']
                s = x['s']
            else:
                valid, N, m, s = self.data_from_threshold(t)
            if valid:
               self.thresholds.append((t, N, m, s))

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
            self.fig.canvas.draw()
        else:
            self.circle = plt.Circle(self.circle_pos, self.circle_radius, color=(1, 0, 0, 0.25))
            self.ax_image.add_artist(self.circle)
        self.centerx.label.set_text("< X: %d >" % self.circle_pos[0])
        self.centery.label.set_text("< Y: %d >" % self.circle_pos[1])
        self.radius.label.set_text("< R: %d >" % self.circle_radius)
        self.fig.canvas.draw()

    def on_prev_clicked(self, ev):
        self.save_data()
        if self.current_index > 0:
            self.load_index(self.current_index - 1)

    def on_next_clicked(self, ev):
        self.save_data()
        if self.current_index < len(self.ids) - 1:
            self.load_index(self.current_index + 1)

    def on_save_clicked(self, ev):
        self.save_data()

    def on_centerx_clicked(self, ev):
        if ev.xdata is None:
            return
        if ev.xdata < 0.5:
            newx = self.circle_pos[0] - INCR_X
            if newx < 0:
                newx = 0
        else:
            newx = self.circle_pos[0] + INCR_X
            if newx > self.background.shape[1]:
                newx = self.background.shape[1]
        if newx != self.circle_pos[0]:
            self.circle_pos = (newx, self.circle_pos[1])
            self.update_image_circle()

    def on_centery_clicked(self, ev):
        if ev.xdata is None:
            return
        if ev.xdata < 0.5:
            newy = self.circle_pos[1] - INCR_Y
            if newy < 0:
                newy = 0
        else:
            newy = self.circle_pos[1] + INCR_Y
            if newy > self.background.shape[0]:
                newy = self.background.shape[0]
        if newy != self.circle_pos[1]:
            self.circle_pos = (self.circle_pos[0], newy)
            self.update_image_circle()

    def on_radius_clicked(self, ev):
        if ev.xdata is None:
            return
        if ev.xdata < 0.5:
            newradius = self.circle_radius - INCR_RADIUS
            if newradius < 0:
                newradius = 0
        else:
            newradius = self.circle_radius + INCR_RADIUS
        if newradius != self.circle_radius:
            self.circle_radius = newradius
            self.update_image_circle()

    def on_button_pressed(self, ev):
        if ev.xdata is None or ev.ydata is None:
            return

        ax = ev.inaxes
        if ax in [self.ax_objects, self.ax_area]:
            if self.current_threshold != ev.xdata:
                self.current_threshold = ev.xdata

                self.line_objects[0].remove()
                self.line_area[0].remove()
                self.line_objects = self.ax_objects.plot((self.current_threshold, self.current_threshold), (-10000, 10000), '--', color='red')
                self.line_area = self.ax_area.plot((self.current_threshold, self.current_threshold), (-1000000, 1000000), '--', color='red')

                self.show_threshold()
                self.fig.canvas.draw()
        elif ax == self.ax_image:
            if ev.button == 3:
                self.mouse_points = []
                self.circle_pos = (ev.xdata, ev.ydata)
                self.update_image_circle()
            else:
                self.mouse_points.append((ev.xdata, ev.ydata))
                if len(self.mouse_points) == 3:
                    center, radius = circle_3pt(*self.mouse_points)
                    self.circle_pos = center
                    self.circle_radius = radius
                    self.update_image_circle()
                    self.mouse_points = []


    def on_button_released(self, ev):
        pass

    def run_plot(self):
        plt.show()

    def precalculate_threshold_data(self):
        thresholds = np.linspace(start=0.00001, stop=0.001, num=30)
        for id in self.ids:
            self.current_id = id
            result = {}
            for t in thresholds:
                valid, N, m, s = self.data_from_threshold(t)
                result[t] = {"valid": valid, "N": N, "m": m, "s": s}
            filename = os.path.join(self.cache_dir, "cache-%s.json" % id)
            with open(filename, "w") as f:
                json.dump(result, f)

if __name__ == '__main__':
    try:
        os.makedirs(PRETREATMENT_DIR)
    except:
        pass
    dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR + d)]
    picker_data_file = os.path.join(PRETREATMENT_DIR, "threshold_picker_data.json")
    ip = InteractivePlot(dirs, picker_data_file, PRETREATMENT_DIR, 0.0005)
    ip.run_plot()
    #ip.precalculate_threshold_data()

    #ex_id = '20130318_131111'
    #threshold = 0.0001
    #threshold = 0.0003
