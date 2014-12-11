#!/usr/bin/env python
import os
import sys
import itertools
import glob
import pandas as pd
import numpy as np

from waldo.conf import settings
from waldo.wio import Experiment
import waldo.wio as wio
import waldo.collider
import multiworm
import waldo.metrics.report_card as report_card


class OutputWriter(object):

    def __init__(self, ex_id, graph=None, data_dir=None, output_dir=None):
        """ """
        if data_dir is None:
            data_dir = str(settings.MWT_DATA_ROOT)
        if output_dir is None:
            #TODO: once output dir is defined in settings, change this to point at that instead of hard coding it.
            output_dir = os.path.abspath('./../data/chore/')

        #print '--------------------------', ex_id, '--------------------------'
        # print(output_dir, ex_id)
        # print(data_dir, ex_id)
        self.ex_id = ex_id
        self.output_dir = os.path.join(output_dir, ex_id)
        self.data_dir  = os.path.join(data_dir, ex_id)
        self.experiment = Experiment(experiment_id=ex_id, data_root=data_dir)

        if graph is None:
            graph_orig = self.experiment.graph.copy()
            graph, _ = report_card.iterative_solver(self.experiment, graph_orig)
        self.graph = graph

    def export(self):
        """

        """

        ex_id = self.ex_id
        print('loading summary data')
        basename, summary_df = self._load_summary()
        lost_and_found = self._create_lost_and_found()
        #json.dump(lost_and_found, open('lost_and_found.json', 'w'))
        #blob_basename = '{bn}'.format(bn=basename)

        # make this run as last step before saving
        wio.file_manager.ensure_dir_exists(self.output_dir)
        print('writing blobs files')
        file_management = self._write_blob_files(basename, summary_df)

        #lost_and_found = json.load(open('lost_and_found.json', 'r'))
        #file_management = json.load(open('file_m.json', 'r'))
        print 'recreating summary file'
        summary_lines = self._recreate_summary_file(summary_df, lost_and_found, file_management)
        # def _recreate_summary_file(self, summary_df, lost_and_found, file_management, basename='chore'):
        sum_name = os.path.join(self.output_dir, '{bn}.summary'.format(bn=basename))
        self._write_summary_lines(sum_name, summary_lines)

    def _write_summary_lines(self, sum_name, lines):
        """
        Keyword Arguments:
        lines --
        """
        print sum_name, 'is writing'
        with open(sum_name, 'w') as f:
            for line in lines:
                f.write(line)

    def format_number_string(self, num, ndigits=5):
        s = str(int(num))
        for i in range(ndigits):
            if len(s) < ndigits:
                s = '0' + s
        return s

    def _write_blob_files(self,  basename='chore', summary_df=None):
        experiment = self.experiment
        graph = self.graph
        #all_frames = [int(f) for f in summary_df.index]
        #sdf = summary_df.reindex(all_frames)
        sdf = summary_df.set_index(0)

        file_management = {}
        #print basename
        file_path = os.path.join(self.output_dir, basename)
        #print file_path

        mashup_history = []
        file_counter = 0 # increments every time file is sucessfully written
        for node in self.graph:
            # get save file name in order.
            #print node
            #print node_file

            # start storing blob index into file_manager dict.
            node_data = graph.node[node]
            died_f = node_data['died_f']
            components = node_data.get('components', [])


            if not components:
                components = [node]
            #print components

            line_data = []
            component_record = []
            for bid in components:
                try:
                    lines = [l for l in experiment._blob_lines(bid)]
                except multiworm.core.MWTDataError:
                    continue
                except ValueError:
                    continue

                for l in lines:
                    l = l.strip()
                    parts = l.split()
                    if len(parts) < 5:
                        continue

                    # line_data.append({'frame': int(parts[0]),
                    #                   'x': float(parts[2]),
                    #                   'y': float(parts[3]),
                    #                   'bid': bid,
                    #                   'line':l})
                    parts[0] = int(parts[0])
                    parts[2] = float(parts[2])
                    parts[3] = float(parts[3])
                    line_data.append(parts)
                    component_record.append(bid)
            if not line_data:
                continue
            compiled_lines = pd.DataFrame(line_data)
            compiled_lines.fillna(' ', inplace=True)
            compiled_lines.rename(columns={0:'frame', 1:'time', 2:'x', 3:'y'},
                                  inplace=True)


            mashup_df = compiled_lines[['frame', 'time', 'x', 'y']]
            mashup_df['node'] = node
            mashup_df['bid'] = component_record
            mashup_df = mashup_df[mashup_df.duplicated('frame')]
            mashup_history.append(mashup_df)

            compiled_lines.sort('frame', inplace=True)
            compiled_lines.drop_duplicates('frame', take_last=False,
                                           inplace=True)


            existing_frames = list(compiled_lines['frame'])
            all_frames = np.arange(int(existing_frames[0]), int(existing_frames[-1]) + 1)
            all_frames = [int(i) for i in all_frames]
            #print all_frames[0], type(all_frames[0])
            if len(existing_frames) != len(all_frames):
                #print len(existing_frames), 'existing'
                #print len(all_frames), 'with gaps filled'
                #print compiled_lines.head(2)

                compiled_lines.set_index('frame', inplace=True)
                compiled_lines = compiled_lines.reindex(all_frames)
                #print compiled_lines.head()
                for f in compiled_lines[compiled_lines['time'].isnull()].index:
                    t = round(sdf.loc[f][1], ndigits=3)
                    #print f, t, 'fill time'
                    compiled_lines['time'].loc[f] = t



                compiled_lines['x'] = compiled_lines['x'].interpolate()
                compiled_lines['y'] = compiled_lines['y'].interpolate()
                for i in range(10, len(compiled_lines.columns)):
                    compiled_lines[i] = compiled_lines[i].fillna(' ')

                cl = compiled_lines.fillna(method='ffill')
                compiled_lines = cl.reset_index()
                #print compiled_lines.head()

            # Now that actual lines of data found for blob
            # Store the data and
            if died_f not in file_management:
                file_management[died_f] = []
            location = '{f}.{pos}'.format(f=file_counter, pos=0)
            file_management[died_f].extend([[node, location]])
            file_number = self.format_number_string(file_counter)
            node_file = '{p}_{n}k.blobs'.format(p=file_path,
                                                n=file_number)
            with open(node_file, 'w') as f:
                f.write('% {n}\n'.format(n=node))

                for j, row in compiled_lines.iterrows():
                    #print row
                    #print row['line']
                    row = ' '.join(['{i}'.format(i=i) for i in row])
                    row = row + '\n'
                    f.write(row)
            file_counter += 1 # increments every time file is sucessfully written
        all_mashups = pd.concat(mashup_history)
        all_mashups.sort('frame', inplace=True)
        all_mashups.to_csv('mashup_record.csv', index=False)
        print file_management
        return file_management

    def _load_summary(self):
        ex_dir = str(self.experiment.directory)
        search_path = os.path.join(ex_dir, '*.summary')
        # print 'summary:', glob.glob(search_path)
        # TODO
        summary_path = glob.glob(search_path)[0]
        basename = os.path.basename(summary_path).split('.summary')[0]
        print basename
        with open(summary_path, 'r') as f:
            lines = f.readlines()

        cleaned_lines = []
        for line in lines:
            line = line.split('%')[0]
            parts = line.split()
            parts[0] = int(parts[0])
            parts[1] = round(float(parts[1]), ndigits=3)
            cleaned_lines.append(parts)
        summary_df = pd.DataFrame(cleaned_lines)
        #print summary_df.head()
        return basename, summary_df

    def _create_lost_and_found(self):
        graph = self.graph
        lost_and_found = {}
        for i, node in enumerate(graph):
            #print node
            node_data = graph.node[node]
            #print node_data
            successors = list(graph.successors(node))
            predecessors = list(graph.predecessors(node))
            s_data = [[node, 0]]
            p_data = [[0, node]]

            if successors:
                s_data =[[node, s] for s in successors]
            if predecessors:
                p_data = [[p, node] for p in predecessors]

            born_f = node_data['born_f']
            died_f = node_data['died_f']

            if died_f not in lost_and_found:
                lost_and_found[died_f] = []
            if born_f not in lost_and_found:
                lost_and_found[born_f] = []

            lost_and_found[died_f].extend(s_data)
            lost_and_found[born_f].extend(p_data)
        return lost_and_found

    def _recreate_summary_file(self, summary_df, lost_and_found, file_management):
        """ writes a summary file based on the data collected
        """
        #summary_df['lost_found'] = ''
        #summary_df['location'] = ''
        lf_lines = {}
        fm_lines = {}
        blobs_in_files = []
        for fl in file_management.values():
            if len(fl) > 1:
                bs, fs = zip(*fl)
                blobs_in_files.extend(bs)
            else:
                bs, fs = fl[0]
                blobs_in_files.append(bs)

        for frame, lf in lost_and_found.iteritems():
            line_contains_info = False
            lf_line = ' %%'
            for a, b in lf:
                if a not in blobs_in_files: # remove references to blobs that are not actually saved
                    a = 0
                if b not in blobs_in_files:
                    b = 0
                if a == 0 and b == 0:
                    continue
                else:
                    lf_line = lf_line + ' {a} {b}'.format(a=int(a), b=int(b))
                    line_contains_info = True

            if line_contains_info: # only save this line if it contains data
                lf_lines[int(frame)] = lf_line
            #print frame, lf_line

        for frame, fm in file_management.iteritems():
            fm_line = ' %%%'
            for a, b in fm:
                fm_line = fm_line + ' {a} {b}'.format(a=a, b=b)
            #print frame, fm_line
            fm_lines[int(frame)] = fm_line

        lines = []
        for i, row in summary_df.iterrows():
            line = ' '.join(['{i}'.format(i=i) for i in row])
            frame = int(row[0])
            #print frame
            line = line + lf_lines.get(frame, '')
            line = line + fm_lines.get(frame, '')
            #if lf_lines.get(frame, ''):
            #    print line
            #if fm_lines.get(frame, ''):
            #    print line
            line = line + '\n'
            lines.append(line)
        #print lost_and_found
        # TODO: add both lines to summary file.
        return lines
