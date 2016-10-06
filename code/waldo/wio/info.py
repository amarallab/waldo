import os
import pathlib
import json
import shutil

# project specific
from waldo.conf import settings
from waldo.wio import paths


INFO_FILE_NAME = "info.json"

def get_initial_info_data(path):
    name = ''
    summary_files = list(pathlib.Path(path).glob('**' + os.sep + '*.summary'))
    if len(summary_files) > 0:
        summary_file = summary_files[0]
        name = os.path.splitext(summary_file.name)[0]
    eid = pathlib.Path(path).name
    camera_id = 0
    try:
        camera_id = eid.split("-")[1]
    except:
        pass
    info = {'name': name,
            'eid': eid,
            'strain': '',
            'camera_id': camera_id,
            'notes': ''}
    return info


def create_and_copy(ex_id, root=None, created_by="", settings_data=None, roi_data=None, calibration_data=None):
    input_path = paths.experiment(ex_id, root=root)
    output_path = paths.waldo_data(ex_id, root=root)

    if settings_data is None:
        settings_data = settings._data

    if roi_data is None:
        annotation_filename = paths.threshold_data(ex_id)
        try:
            with open(str(annotation_filename), "rt") as f:
                roi_data = json.loads(f.read())
        except IOError as ex:
            roi_data = {}

    if calibration_data is None:
        calibration_filename = paths.calibration_data(ex_id)
        try:
            with open(str(calibration_filename), "rt") as f:
                calibration_data = json.loads(f.read())
        except IOError as ex:
            calibration_data = {}

    try:
        input_file_name = input_path / INFO_FILE_NAME
        if not input_file_name.is_file():
            info = get_initial_info_data(input_path)
        else:
            with open(str(input_file_name), 'r') as f:
                info = json.loads(f.read())
        info['created_by'] = created_by
        info['settings'] = settings_data
        info['roi'] = roi_data
        info['calibration'] = calibration_data
        output_file_name = output_path / INFO_FILE_NAME
        with open(str(output_file_name), 'w') as f:
            f.write(json.dumps(info)) 
    except IOError:
        print("WARNING: info file not created.")
