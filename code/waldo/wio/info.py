import os
import pathlib
import json
import shutil
from . import paths

INFO_FILE_NAME = "info.json"

def create(path, created_by=""):
    """creates an info.json file"""
    try:
        info_file_name = path / INFO_FILE_NAME
        if not info_file_name.is_file():
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
                    'created_by': created_by,
                    'notes': ''}
            with open(str(info_file_name), 'w') as outfile:
                json.dump(info, outfile)
                outfile.close()
    except IOError:
        print("WARNING: info file not created.")


def create_and_copy(ex_id, root=None, created_by="guiwaldo.py"):
    print('-------------------------------------Create and Copy!')
    input_path = paths.experiment(ex_id, root=root)
    output_path = paths.waldo_data(ex_id, root=root)
    print(output_path)
    info_file_name = output_path / INFO_FILE_NAME
    print(info_file_name)
    print(info_file_name.is_file())
    # if not os.path.exists(input_path + os.sep + INFO_FILE_NAME):
    if not info_file_name.is_file():
        create(input_path, created_by)
    try:
        shutil.copy(str(input_path) + os.sep + INFO_FILE_NAME, str(output_path))
    except IOError:
        print("WARNING: info file not created.")
