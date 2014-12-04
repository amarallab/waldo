import pathcustomize
from waldo.output.writer import OutputWriter

def main(ex_id = '20130318_131111'):
    print '--------------------------', ex_id, '--------------------------'
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
          #'20130323_170503', # moe - failed. back in time.
          '20130323_143707', # larry
          '20130323_132240', # larry
          '20130323_143706', # curly
          '20130324_130602'] # curly

ex_ids = ['20130702_135704']
if __name__ == '__main__':
    for ex_id in ex_ids[:]:
        main(ex_id)
#main('20130702_135704')
#main('20141017_113435')
