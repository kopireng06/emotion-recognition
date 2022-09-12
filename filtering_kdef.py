import os
import fnmatch
import shutil
from pathlib import Path

expression_codes = {
    'ANS': 'angry',
    'AFS': 'fear',
    'DIS': 'disgusted',
    'HAS': 'happy',
    'NES': 'neutral',
    'SAS': 'sad',
    'SUS': 'surprised'
}


def copy_image(src_path, expression_code, destination_path="/"):
    abs_path = os.path.abspath("")
    if(expression_code in expression_codes):
        try:
            os.makedirs(abs_path+'/'+destination_path+'/' +
                        expression_codes.get(expression_code))
        except:
            print("Directory Already Exists!")
        destination_to_copy = Path(
            destination_path+'/'+expression_codes.get(expression_code))
        shutil.copy2(src_path, destination_to_copy)


def get_expression_code(file_name):
    file_name_without_mime = file_name.split('.')[0]
    expression_code = file_name_without_mime[len(
        file_name_without_mime)-3:len(file_name_without_mime)]
    return expression_code


for dirpath, dirnames, files in os.walk('./KDEF_and_AKDEF/KDEF_and_AKDEF/KDEF'):
    for file_name in files:
        expression_code = get_expression_code(file_name)
        copy_image(dirpath+'/'+file_name, expression_code, "KDEF")
