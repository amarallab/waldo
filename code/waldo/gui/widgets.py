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
import matplotlib.patches as patches

# from waldo.images.grab_images import grab_images_in_time_range
from waldo.gui import tasking
from waldo.wio import paths
from .loaders import CacheThresholdLoadingDialog
from .helpers import circle_3pt

from waldo.conf import settings
from waldo.prepare import summarize as prepare_summarize
from waldo.images import summarize as images_summarize
from waldo.metrics.report_card import WaldoSolver
from waldo.wio import Experiment
from waldo.output.writer import OutputWriter
from waldo import wio
import waldo.images.evaluate_acuracy as ea
import waldo.images.worm_finder as wf
import waldo.metrics.report_card as report_card
import waldo.metrics.step_simulation as ssim
import waldo.viz.eye_plots as ep
from waldo.gui import pathcustomize


STYLE = 'ggplot'

try:
    # MPL 1.4+
    plt.style.use(STYLE)
except AttributeError:
    # fallback to mpltools
    from mpltools import style
    style.use(STYLE)


class ThresholdCacheWidget(QtGui.QWidget):
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

        layout = QtGui.QGridLayout()

        q1 = QtGui.QLabel("<b>Choose Threshold</b>")
        layout.addWidget(q1, 0, 0, 1, 1)
        layout.addWidget(QtGui.QLabel("Click on either graph to pick a threshold value"), 1, 0, 1, 1)
        layout.addWidget(self.histogram_canvas, 2, 0, 1, 1)
        layout.addWidget(self.histogram_toolbar, 3, 0, 1, 1)

        q2 = QtGui.QLabel("<b>Define Region of Interest</b>")
        layout.addWidget(q2, 0, 1, 1, 1)
        layout.addWidget(QtGui.QLabel("Click on image three times to define the region of interest"), 1, 1, 1, 1)
        layout.addWidget(self.image_canvas, 2, 1, 1, 1)
        layout.addWidget(self.image_toolbar, 3, 1, 1, 1)
        self.setLayout(layout)

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




class ExperimentResultWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(ExperimentResultWidget, self).__init__()

        self.tab = QtGui.QTabWidget()

        # Figure 1
        figure = plt.figure()
        canvas = FigureCanvas(figure)
        canvas.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        canvas.setMinimumSize(50, 50)
        toolbar = NavigationToolbar(canvas, self)
        toolbar.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        widget = QtGui.QWidget()
        layout = QtGui.QVBoxLayout()
        layout.addWidget(canvas)
        layout.addWidget(toolbar)
        widget.setLayout(layout)
        self.tab.addTab(widget, 'Results')
        self.plot = figure

        # Tab 1
        self.trackCountsTable = QtGui.QTableWidget()
        self.tab.addTab(self.trackCountsTable, 'Track Counts')

        # Tab 2
        self.networkOverviewTable = QtGui.QTableWidget()
        self.tab.addTab(self.networkOverviewTable, 'Network Overview')

        # Tab 3
        self.trackTerminationTable = QtGui.QTableWidget()
        self.trackInitiationTable = QtGui.QTableWidget()

        widget = QtGui.QWidget()
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.trackTerminationTable)
        layout.addWidget(self.trackInitiationTable)
        widget.setLayout(layout)
        self.tab.addTab(widget, 'Track Fragmentation')

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.tab)
        self.setLayout(layout)

    def initializeWidget(self, experiment):
        self.make_figures(experiment.id)

        report_card_df = experiment.prepdata.load('report-card')
        if report_card_df is None:
            self.trackCountsTable.clear()
            self.networkOverviewTable.clear()
        else:
            self.make_trackCountsTable(report_card_df)
            self.make_networkOverviewTable(report_card_df)

        end_report_df = experiment.prepdata.load('end_report')
        if end_report_df is None:
            self.trackTerminationTable.clear()
        else:
            self.make_trackTerminationTable(end_report_df)

        start_report_df = experiment.prepdata.load('start_report')
        if start_report_df is None:
            self.trackInitiationTable.clear()
        else:
            self.make_trackInitiationTable(start_report_df)


    def make_trackCountsTable(self, report_card_df):
        headers = ['phase', 'step', 'total-nodes', '>10min', '>20min', '>30min', '>40min', '>50min', 'duration-mean', 'duration-std']
        rightAligns = [False, False, True, True, True, True, True, True, True, True ]
        widths = [100, 150, 100, 100, 100, 100, 100, 100, 100, 100]

        self.trackCountsTable.clear()
        self.trackCountsTable.setColumnCount(len(headers))
        self.trackCountsTable.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.trackCountsTable.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        for col, (header, width) in enumerate(zip(headers, widths)):
            self.trackCountsTable.setHorizontalHeaderItem(col, QtGui.QTableWidgetItem(header))
            self.trackCountsTable.setColumnWidth(col, width)
        self.trackCountsTable.verticalHeader().setVisible(False)

        prev_phase = ''
        b = report_card_df[headers]
        for row_data in b.iterrows():
            row = row_data[0]
            self.trackCountsTable.setRowCount(row + 1)
            col_data = row_data[1]
            for col, (header, rightAlign) in enumerate(zip(headers, rightAligns)):
                current = str(col_data[header])
                if col == 0 and current == prev_phase:
                    current = ""
                item = QtGui.QTableWidgetItem(current)
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                if rightAlign:
                    item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
                self.trackCountsTable.setItem(row, col, item)
            prev_phase = col_data['phase']

    def make_networkOverviewTable(self, report_card_df):
        headers = ['phase', 'step', 'total-nodes', 'connected-nodes', 'isolated-nodes', 'giant-component-size', '# components']
        rightAligns = [False, False, True, True, True, True, True]
        widths = [100, 150, 100, 100, 100, 100, 100]

        self.networkOverviewTable.clear()
        self.networkOverviewTable.setColumnCount(len(headers))
        self.networkOverviewTable.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.networkOverviewTable.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        for col, (header, width) in enumerate(zip(headers, widths)):
            self.networkOverviewTable.setHorizontalHeaderItem(col, QtGui.QTableWidgetItem(header))
            self.networkOverviewTable.setColumnWidth(col, width)
        self.networkOverviewTable.verticalHeader().setVisible(False)

        prev_phase = ''
        b = report_card_df[headers]
        for row_data in b.iterrows():
            row = row_data[0]
            self.networkOverviewTable.setRowCount(row + 1)
            col_data = row_data[1]
            for col, (header, rightAlign) in enumerate(zip(headers, rightAligns)):
                current = str(col_data[header])
                if col == 0 and current == prev_phase:
                    current = ""
                item = QtGui.QTableWidgetItem(current)
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                if rightAlign:
                    item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
                self.networkOverviewTable.setItem(row, col, item)
            prev_phase = col_data['phase']

    def make_trackTerminationTable(self, end_report_df):
        df = end_report_df
        df['lifespan'] = ['0-1 min', '1-5 min', '6-10 min', '11-20 min', '21-60 min', 'total']
        df = df.rename(columns={'lifespan': 'track-duration',
                                'unknown': 'disappear',
                                'timing':'recording-finishes',
                                'on_edge':'image-edge'})
        df.set_index('track-duration', inplace=True)

        headers = ['', 'disappear', 'split', 'join', 'recording-finishes', 'image-edge', 'outside-roi']
        rightAligns = [False, True, True, True, True, True, True]
        widths = [100, 100, 100, 100, 100, 100, 100]

        self.trackTerminationTable.clear()
        self.trackTerminationTable.setColumnCount(len(headers))
        self.trackTerminationTable.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.trackTerminationTable.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        for col, (header, width) in enumerate(zip(headers, widths)):
            self.trackTerminationTable.setHorizontalHeaderItem(col, QtGui.QTableWidgetItem(header))
            self.trackTerminationTable.setColumnWidth(col, width)
        self.trackTerminationTable.verticalHeader().setVisible(False)

        b = df[headers[1:]]
        for row, row_data in enumerate(b.iterrows()):
            first_label = row_data[0]
            self.trackTerminationTable.setRowCount(row + 1)
            col_data = row_data[1]
            for col, (header, rightAlign) in enumerate(zip(headers, rightAligns)):
                if col == 0:
                    current = first_label
                else:
                    current = str(col_data[header])
                item = QtGui.QTableWidgetItem(current)
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                if rightAlign:
                    item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
                self.trackTerminationTable.setItem(row, col, item)

    def make_trackInitiationTable(self, start_report_df):
        df = start_report_df
        df['lifespan'] = ['0-1 min', '1-5 min', '6-10 min', '11-20 min', '21-60 min', 'total']
        df = df.rename(columns={'lifespan': 'track-duration',
                                'unknown': 'appear',
                                'timing':'recording-begins',
                                'on_edge':'image-edge'})
        df.set_index('track-duration', inplace=True)

        headers = ['', 'appear', 'split', 'join', 'recording-begins', 'image-edge', 'outside-roi']
        rightAligns = [False, True, True, True, True, True, True]
        widths = [100, 100, 100, 100, 100, 100, 100]

        self.trackInitiationTable.clear()
        self.trackInitiationTable.setColumnCount(len(headers))
        self.trackInitiationTable.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.trackInitiationTable.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        for col, (header, width) in enumerate(zip(headers, widths)):
            self.trackInitiationTable.setHorizontalHeaderItem(col, QtGui.QTableWidgetItem(header))
            self.trackInitiationTable.setColumnWidth(col, width)
        self.trackInitiationTable.verticalHeader().setVisible(False)

        b = df[headers[1:]]
        for row, row_data in enumerate(b.iterrows()):
            first_label = row_data[0]
            self.trackInitiationTable.setRowCount(row + 1)
            col_data = row_data[1]
            for col, (header, rightAlign) in enumerate(zip(headers, rightAligns)):
                if col == 0:
                    current = first_label
                else:
                    current = str(col_data[header])
                item = QtGui.QTableWidgetItem(current)
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                if rightAlign:
                    item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
                self.trackInitiationTable.setItem(row, col, item)

    # MAKE FIG

    def calculate_stats_for_moves(self, min_moves, prep_data):
        move = prep_data.load('moved')

        base_accuracy = prep_data.load('accuracy', index_col=False)
        print(base_accuracy.head())
        print(base_accuracy.columns)
        matches = prep_data.load('matches')
        counts, tps, fps, fns = [], [], [], []
        for mm in min_moves:
            print(mm)
            moved = move[move['bl_moved'] >= mm]
            bids = list(moved['bid'])
            filtered_accuracy = ea.recalculate_accuracy(matches, base_accuracy, bids=bids)
            #filtered_accuracy = filtered_accuracy[np.isfinite(filtered_accuracy['true-pos'])]
            counts.append(len(bids))
            tp = filtered_accuracy['true-pos'].mean()
            #print('tp', tp)
            fp = filtered_accuracy['false-pos'].mean()
            #print('fp', fp)
            fn = filtered_accuracy['false-neg'].mean()

            tps.append(tp)
            fps.append(fp)
            fns.append(fn)

        tps = np.array(tps)
        fps = np.array(fps)
        fns = np.array(fns)
        totals = fns + tps
        tps_p = tps / totals * 100
        fps_p = fps / totals * 100
        fns_p = fns / totals * 100

        print('true counts=', np.mean(totals))

        data = pd.DataFrame([tps_p, fns_p, fps_p], index=['TP', 'FN', 'FP']).T
        counts = pd.DataFrame(counts, columns=['counts'])
        return data, counts

    def _new_plot(self, ax, df, label, nth_color=0, xmax=60):
        step_df = df
        n_steps = len(step_df)
        steps = []

        xs = list(step_df['t0'])
        widths = list(step_df['lifespan'])
        height = 1

        color = ax._get_lines.color_cycle.next()
        for i in range(nth_color):
            color = ax._get_lines.color_cycle.next()

        for y, (x, width) in enumerate(zip(xs, widths)):
            steps.append(patches.Rectangle((x,y), height=height, width=width,
                                           fill=True, fc=color, ec=color, alpha=0.5))
        for step in steps:
            ax.add_patch(step)

        ax.plot([0], color=color, label=label, alpha=0.5)
        ax.set_xlim([0, xmax])

    def steps_from_node_report(self, experiment, min_bl=1):
        node_report = experiment.prepdata.load('node-summary')
        print(node_report.head())
        steps = node_report[['bl', 't0', 'tN', 'bid']]
        steps.set_index('bid', inplace=True)

        steps['t0'] = steps['t0'] / 60.0
        steps['tN'] = steps['tN'] / 60.0
        steps['lifespan'] = steps['tN'] - steps['t0']
        steps['mid'] = (steps['tN']  + steps['t0']) / 2.0
        return steps[steps['bl'] >= 1]

    def make_figures(self, ex_id, step_min_move = 1):
        experiment = wio.Experiment(experiment_id=ex_id)
        graph = experiment.graph.copy()

        self.plot.clf()

        moving_nodes =  [int(i) for i in graph.compound_bl_filter(experiment, threshold=step_min_move)]
        steps, durations = report_card.calculate_duration_data_from_graph(experiment, graph, moving_nodes)

        final_steps = self.steps_from_node_report(experiment)
        accuracy = experiment.prepdata.load('accuracy')
        worm_count = np.mean(accuracy['true-pos'] + accuracy['false-neg'])
        print('worm count:', worm_count)

        ### AX 0
        # print('starting ax 0')
        # color_cycle = ax._get_lines.color_cycle
        # for i in range(5):
        #     c = color_cycle.next()
        # wf.draw_minimal_colors_on_image_T(ex_id, time=30*30, ax=ax, color=c)
        self.step_facets(steps, final_steps)

    def step_facets(self, df, df2, t1=5, t2=20):
        xmax = max([max(df['tN']), max(df['tN'])])

        gs = grd.GridSpec(3, 2)
        gs.update(wspace=0.01, hspace=0.05) #, bottom=0.1, top=0.7)

        ax_l0 = self.plot.add_subplot(gs[0,0])
        ax_l1 = self.plot.add_subplot(gs[0,1], sharey=ax_l0)
        ax_m0 = self.plot.add_subplot(gs[1,0])
        ax_m1 = self.plot.add_subplot(gs[1,1], sharey=ax_m0)
        ax_s0 = self.plot.add_subplot(gs[2,0])
        ax_s1 = self.plot.add_subplot(gs[2,1], sharey=ax_s0)

        left = [ax_l0, ax_m0, ax_s0]
        right = [ax_l1, ax_m1, ax_s1]
        top = [ax_l0, ax_l1]
        mid = [ax_m0, ax_m1]
        bottom = [ax_s0, ax_s1]
        ylabels = ['> {t2} min'.format(t2=t2), '{t1} to {t2} min'.format(t1=t1, t2=t2), '< {t1} min'.format(t1=t1)]

        short1 = df[df['lifespan'] < t1]
        short2 = df2[df2['lifespan'] < t1]
        mid1 = df[(df['lifespan'] < t2) & (df['lifespan'] >= t1)]
        mid2 = df2[(df2['lifespan'] < t2) & (df2['lifespan'] >= t1)]
        long1 = df[df['lifespan'] >= t2]
        long2 = df2[df2['lifespan'] >= t2]

        short_max = max([len(short1), len(short2)])
        mid_max = max([len(mid1), len(mid2)])
        long_max = max([len(long1), len(long2)])
        print('short', short_max)
        print('long', long_max)
        print('mid', mid_max)
        print(long1.head(20))

        self._new_plot(ax_s0, short1, 'raw', xmax=xmax)
        self._new_plot(ax_m0, mid1, 'raw', xmax=xmax)
        self._new_plot(ax_l0, long1, 'raw', xmax=xmax)

        self._new_plot(ax_s1, short2, 'final', nth_color=1, xmax=xmax)
        self._new_plot(ax_m1, mid2, 'final', nth_color=1, xmax=xmax)
        self._new_plot(ax_l1, long2, 'final', nth_color=1, xmax=xmax)

        for ax in right:
            ax.axes.yaxis.tick_right()

        for ax, t in zip(top, ['raw', 'final']):
            ax.get_xaxis().set_ticklabels([])
            ax.set_ylim([0, long_max])

        for ax in mid:
            ax.get_xaxis().set_ticklabels([])
            ax.set_ylim([0, mid_max])

        for ax in bottom:
            ax.set_xlabel('t (min)')
            ax.set_ylim([0, short_max])

        for ax, t in zip(left, ylabels):
            ax.set_ylabel(t)

        ax_s0.get_xaxis().set_ticklabels([0, 10, 20, 30, 40, 50])
