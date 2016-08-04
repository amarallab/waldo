#!/usr/bin/env python
from __future__ import absolute_import, print_function

# standard library
import pathlib

# third party
import pandas as pd
import numpy as np

# project specific
# from waldo.conf import settings
from waldo import wio
from waldo.wio import paths
import multiworm
import waldo.metrics.report_card as report_card


class OutputWriter(object):
    """

    """
    def __init__(self, ex_id, graph=None, data_dir=None, output_dir=None):
        self.ex_id = ex_id

        if data_dir is None:
            self.data_dir = paths.experiment(ex_id)
        else:
            self.data_dir = pathlib.Path(data_dir)

        if output_dir is None:
            self.output_dir = paths.output(ex_id)
        else:
            self.output_dir = pathlib.Path(output_dir)

        self.experiment = wio.Experiment(fullpath=self.data_dir)

        if graph is None:
            graph_orig = self.experiment.graph.copy()
            graph, _ = report_card.iterative_solver(self.experiment, graph_orig)
        self.graph = graph

    def export(self, interpolate=False, callback1=None, callback2=None):
        """

        """

        ex_id = self.ex_id
        print('loading summary data')
        basename, summary_df = self._load_summary()
        lost_and_found = self._create_lost_and_found()
        # json.dump(lost_and_found, open('lost_and_found.json', 'w'))
        # blob_basename = '{bn}'.format(bn=basename)

        # make this run as last step before saving
        paths.mkdirp(self.output_dir)
        print('writing blobs files')
        file_management = self._write_blob_files(basename=basename,
                                                 summary_df=summary_df,
                                                 callback=callback1,
                                                 interpolate=interpolate)

        # lost_and_found = json.load(open('lost_and_found.json', 'r'))
        # file_management = json.load(open('file_m.json', 'r'))
        print('recreating summary file')
        summary_lines = self._recreate_summary_file(summary_df, lost_and_found, file_management, callback=callback2)
        # def _recreate_summary_file(self, summary_df, lost_and_found, file_management, basename='chore'):
        sum_name = self.output_dir / self.experiment.summary_file.name
        self._write_summary_lines(sum_name, summary_lines)

    def _write_summary_lines(self, sum_name, lines):
        """
        Keyword Arguments:
        lines --
        """
        print(sum_name, 'is writing')
        with open(str(sum_name), 'w') as f:
            for line in lines:
                f.write(line)

    def format_number_string(self, num, ndigits=5):
        s = str(int(num))
        for i in range(ndigits):
            if len(s) < ndigits:
                s = '0' + s
        return s

    def _write_blob_files(self, basename='chore', interpolate='False',
                          summary_df=None, callback=None):
        experiment = self.experiment
        graph = self.graph
        # all_frames = [int(f) for f in summary_df.index]
        # sdf = summary_df.reindex(all_frames)

        sdf = summary_df.set_index(0)

        file_management = {}

        mashup_history = []
        file_counter = 0  # increments every time file is sucessfully written
        node_length = len(self.graph)
        for current_node_index, node in enumerate(self.graph):
            # get save file name in order.
            # print(node)
            # print(node_file)

            if callback:
                callback(current_node_index / float(node_length))
            # start storing blob index into file_manager dict.
            node_data = graph.node[node]
            died_f = node_data['died_f']
            components = node_data.get('components', [])

            if not components:
                components = [node]
            # print(components)

            line_data = []
            component_record = []
            for bid in components:
                try:
                    lines = [l for l in experiment._blob_lines(int(bid))]
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
            compiled_lines.rename(columns={0: 'frame', 1: 'time', 2: 'x', 3: 'y'},
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
            # print(all_frames[0], type(all_frames[0]))

            # If interpolate == True,
            # this section inserts values into otherwise empty rows
            if interpolate and len(existing_frames) != len(all_frames):
                # print(len(existing_frames), 'existing')
                # print(len(all_frames), 'with gaps filled')
                # print(compiled_lines.head(2))

                compiled_lines.set_index('frame', inplace=True)
                compiled_lines = compiled_lines.reindex(all_frames)
                # print(compiled_lines.head())
                for f in compiled_lines[compiled_lines['time'].isnull()].index:
                    t = round(sdf.loc[f][1], ndigits=3)
                    # print(f, t, 'fill time')
                    compiled_lines['time'].loc[f] = t

                compiled_lines['x'] = compiled_lines['x'].interpolate()
                compiled_lines['y'] = compiled_lines['y'].interpolate()
                for i in range(10, len(compiled_lines.columns)):
                    compiled_lines[i] = compiled_lines[i].fillna(' ')

                cl = compiled_lines.fillna(method='ffill')
                compiled_lines = cl.reset_index()
                # print(compiled_lines.head())

            # Now that actual lines of data found for blob
            # Store the data and
            if died_f not in file_management:
                file_management[died_f] = []
            location = '{f}.{pos}'.format(f=file_counter, pos=0)
            file_management[died_f].extend([[node, location]])
            file_number = self.format_number_string(file_counter)
            node_file = self.output_dir / '{p}_{n}k.blobs'.format(p=self.experiment.basename, n=file_number)
            with open(str(node_file), 'w') as f:
                f.write('% {n}\n'.format(n=node))

                for j, row in compiled_lines.iterrows():
                    # print(row)
                    # print(row['line'])
                    row = ' '.join(['{i}'.format(i=i) for i in row])
                    row = row + '\n'
                    f.write(row)
            file_counter += 1  # increments every time file is sucessfully written
        all_mashups = pd.concat(mashup_history)
        all_mashups.sort('frame', inplace=True)
        all_mashups.to_csv('mashup_record.csv', index=False)
        if callback:
            callback(1.0)
        print(file_management)
        return file_management

    def _load_summary(self):
        basename = self.experiment.basename
        print(basename)
        with self.experiment.summary_file.open() as f:
            lines = f.readlines()
        print('summary file has', len(lines))

        cleaned_lines = []
        for line in lines:
            line = line.split('%')[0]
            parts = line.split()
            parts[0] = int(parts[0])
            parts[1] = round(float(parts[1]), ndigits=3)
            cleaned_lines.append(parts)
        summary_df = pd.DataFrame(cleaned_lines)
        print('summary df has', len(summary_df))
        # print(summary_df.head())
        return basename, summary_df

    def _create_lost_and_found(self):
        graph = self.graph
        lost_and_found = {}
        for i, node in enumerate(graph):

            node_data = graph.node[node]
            # print(node_data)
            successors = list(graph.successors(node))
            predecessors = list(graph.predecessors(node))

            born_f = node_data['born_f']
            died_f = node_data['died_f']

            # only Handle De-Novo Births
            if predecessors:
                # the predecessor node will insert this data
                pass
            else:
                # need to insert this data ourselves
                if born_f not in lost_and_found:
                    lost_and_found[born_f] = []

                p_data = [[0, node]]
                lost_and_found[born_f].extend(p_data)

            # Always Handle Death
            if died_f not in lost_and_found:
                lost_and_found[died_f] = []
            s_data = [[node, 0]]
            if successors:
                s_data = [[node, s] for s in successors]
            lost_and_found[died_f].extend(s_data)
        return lost_and_found

    def _recreate_summary_file(self, summary_df, lost_and_found, file_management, callback=None):
        """ writes a summary file based on the data collected
        """
        # summary_df['lost_found'] = ''
        # summary_df['location'] = ''
        lf_lines = {}
        fm_lines = {}
        blobs_in_files = []

        if callback:
            STEP1_FRAC = 0.25
            STEP2_FRAC = 0.5
            STEP3_FRAC = 0.25

            def cb_step1(p):
                callback(STEP1_FRAC * p)

            def cb_step2(p):
                callback(STEP1_FRAC + STEP2_FRAC * p)

            def cb_step3(p):
                callback(STEP1_FRAC + STEP2_FRAC + STEP3_FRAC * p)
        else:
            def cb_step1(p):
                pass

            cb_step2 = cb_step3 = cb_step1

        count = len(file_management.values())
        for current, fl in enumerate(file_management.values()):
            cb_step1(current / float(count))
            if len(fl) > 1:
                bs, fs = zip(*fl)
                blobs_in_files.extend(bs)
            else:
                bs, fs = fl[0]
                blobs_in_files.append(bs)

        count = len(lost_and_found.items())
        current = 0
        for frame in lost_and_found:
            lf = lost_and_found[frame]
            cb_step2(current / float(count))
            current += 1

            line_contains_info = False
            lf_line = ' %%'
            parents, children = zip(*lf)
            for a, b in lf:
                if a not in blobs_in_files:  # remove references to blobs that are not actually saved
                    a = 0
                if b not in blobs_in_files:
                    b = 0
                if a == 0 and b == 0:
                    continue
                else:
                    lf_line = lf_line + ' {a} {b}'.format(a=int(a), b=int(b))
                    line_contains_info = True

            if line_contains_info:  # only save this line if it contains data
                lf_lines[int(frame)] = lf_line
                # print(frame, lf_line)

        for frame, fm in file_management.iteritems():
            fm_line = ' %%%'
            for a, b in fm:
                fm_line = fm_line + ' {a} {b}'.format(a=a, b=b)
            # print(frame, fm_line)
            fm_lines[int(frame)] = fm_line

        count = summary_df[0].count()
        current = 0
        lines = []
        for i, row in summary_df.iterrows():
            cb_step3(current / float(count))
            current += 1

            line = ' '.join(['{i}'.format(i=i) for i in row])
            frame = int(row[0])
            # print(frame)
            line = line + lf_lines.get(frame, '')
            line = line + fm_lines.get(frame, '')
            # if lf_lines.get(frame, ''):
            #    print(line)
            # if fm_lines.get(frame, ''):
            #    print(line)
            line = line + '\n'
            lines.append(line)
        # print(lost_and_found)
        # TODO: add both lines to summary file.
        return lines
