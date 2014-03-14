def format_dset_summary_name(data_type, dataset, sum_type, dset_dir=None):
    filename = format_filename(ID=dataset, ID_type='dset',
                               data_type=data_type
                               file_tag=sum_type,
                               file_dir=dset_dir,
                               file_type='json')
    return filename

def write_dset_summary(data, data_type, dataset, sum_type, dset_dir=None):
    filename = format_dset_summary_name(data_type, dataset, sum_type, dset_dir)
    json.dump(data, open(filename, 'w'))

def read_dset_summary(data_type, dataset, sum_type='basic', dset_dir=None):
    filename = format_dset_summary_name(data_type, dataset, sum_type, dset_dir)
    return json.load(open(filename, 'r'))
