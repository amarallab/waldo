#!/usr/bin/env python
import os
import sys
import itertools
import glob
import pandas as pd
import numpy as np


import pathcustomize
from waldo.conf import settings
from waldo.wio import Experiment
import waldo.wio as wio

import waldo.collider
import multiworm
import waldo.metrics.report_card as report_card

DATA_DIR = settings.LOGISTICS['filesystem_data']
CHORE_DIR = os.path.abspath('./../data/chore/')
print DATA_DIR

class OutputWriter(object):

    def __init__(self, ex_id, graph=None):
        """ """
        #print '--------------------------', ex_id, '--------------------------'
        self.chore_dir = os.path.join(CHORE_DIR, ex_id)
        self.data_dir  = os.path.join(DATA_DIR, ex_id)
        self.experiment = Experiment(experiment_id=ex_id, data_root=DATA_DIR)

        if graph is None:
            graph_orig = self.experiment.graph.copy()
            graph, _ = report_card.collision_iteration2(self.experiment, graph_orig)
        self.graph = graph

    def export(self):
        """

        """

        basename, summary_df = self._load_summary()
        lost_and_found = self._create_lost_and_found()
        #json.dump(lost_and_found, open('lost_and_found.json', 'w'))
        blob_basename = '{eid}/{bn}'.format(eid=ex_id, bn=basename)

        # make this run as last step before saving
        wio.file_manager.ensure_dir_exists(self.chore_dir)
        file_management = self.write_blob_files(self.graph, self.experiment, blob_basename, summary_df)

        #lost_and_found = json.load(open('lost_and_found.json', 'r'))
        #file_management = json.load(open('file_m.json', 'r'))
        print 'recreating summary file'
        summary_lines = self._recreate_summary_file(summary_df, lost_and_found, file_management)
        # def _recreate_summary_file(self, summary_df, lost_and_found, file_management, basename='chore'):
        sum_name = os.path.join(self.chore_dir, '{bn}.summary'.format(bn=basename))
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
        file_path = os.path.join(CHORE_DIR, basename)
        #print file_path

        mashup_history = []

        for i, node in enumerate(self, graph):
            # get save file name in order.
            node_file = '{p}_{n}k.blobs'.format(p=file_path,
                                                n=self.format_number_string(i))
            #print node
            #print node_file

            # start storing blob index into file_manager dict.
            node_data = graph.node[node]
            died_f = node_data['died_f']
            if died_f not in file_management:
                file_management[died_f] = []
            location = '{f}.{pos}'.format(f=i, pos=0)
            file_management[died_f].extend([[node, location]])
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

            with open(node_file, 'w') as f:
                f.write('% {n}\n'.format(n=node))

                for j, row in compiled_lines.iterrows():
                    #print row
                    #print row['line']
                    row = ' '.join(['{i}'.format(i=i) for i in row])
                    row = row + '\n'
                    f.write(row)

        all_mashups = pd.concat(mashup_history)
        all_mashups.sort('frame', inplace=True)
        all_mashups.to_csv('mashup_record.csv', index=False)
        print file_management
        return file_management

    def _load_summary(self):
        ex_dir = self.experiment.directory
        search_path = os.path.join(ex_dir, '*.summary')
        print 'summary:', glob.glob(search_path)
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

    def _recreate_summary_file(self, summary_df, lost_and_found, file_management, basename='chore'):
        #summary_df['lost_found'] = ''
        #summary_df['location'] = ''
        lf_lines = {}
        fm_lines = {}

        for frame, lf in lost_and_found.iteritems():
            lf_line = ' %%'
            for a, b in lf:
                lf_line = lf_line + ' {a} {b}'.format(a=int(a), b=int(b))
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

def main(ex_id = '20130318_131111'):
    print '--------------------------', ex_id, '--------------------------'
    print CHORE_DIR
    out_writer = OutputWriter(ex_id)
    out_writer.export()
    # chore_dir = os.path.join(CHORE_DIR, ex_id)
    # data_dir  = os.path.join(DATA_DIR, ex_id)
    # print chore_dir
    # wio.file_manager.ensure_dir_exists(chore_dir)
    # experiment = Experiment(experiment_id=ex_id, data_root=DATA_DIR)

    # graph2 = get_graph(experiment)
    # basename, summary_df = load_summary(data_dir)

    # #basename, summary_df = [], []

    # lost_and_found = create_lost_and_found(graph2)
    # #json.dump(lost_and_found, open('lost_and_found.json', 'w'))
    # blob_basename = '{eid}/{bn}'.format(eid=ex_id, bn=basename)
    # file_management = write_blob_files(graph2, experiment, blob_basename, summary_df)
    # #json.dump(file_management, open('file_m.json', 'w'))

    # #lost_and_found = json.load(open('lost_and_found.json', 'r'))
    # #file_management = json.load(open('file_m.json', 'r'))
    # print 'recreating summary file'
    # lines = recreate_summary_file(summary_df, lost_and_found, file_management)
    # sum_name = os.path.join(chore_dir, '{bn}.summary'.format(bn=basename))
    # print sum_name, 'is writing'
    # with open(sum_name, 'w') as f:
    #     for line in lines:
    #         f.write(line)

ex_ids = ['20141017_113435', '20141017_113439', '20141017_123722']

# 20130323_110247 20130323_170503 20130323_143707 20130323_132240 20130323_143706 20130324_130602
ex_ids = ['20130323_110247', # moe
          '20130323_170503', # moe - failed. back in time.
          '20130323_143707', # larry
          '20130323_132240', # larry
          '20130323_143706', # curly - fail
          '20130324_130602'] # curly - fail

if __name__ == '__main__':
    for ex_id in ex_ids[-1:]:
        main(ex_id)
#main('20130702_135704')
#main('20141017_113435')
