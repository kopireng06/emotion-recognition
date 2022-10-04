import os
import fnmatch
import shutil
from pathlib import Path

KDEF_original_path = './KDEF_ORIGINAL/test'
KDEF_filtered_path = 'KDEF_test'

expression_codes = {
    'ANS': 'angry',
    'ANHL': 'angry',
    'ANHR': 'angry',
    'AFS': 'fear',
    'AFHL': 'fear',
    'AFHR': 'fear',
    'DIS': 'disgust',
    'DIHL': 'disgust',
    'DIHR': 'disgust',
    'HAS': 'happy',
    'HAHL:': 'happy',
    'HAHR': 'happy',
    'NES': 'neutral',
    'NEHL': 'neutral',
    'NEHR': 'neutral',
    'SAS': 'sad',
    'SAHL': 'sad',
    'SAHR': 'sad',
    'SUS': 'surprised',
    'SUHL': 'surprised',
    'SUHR': 'surprised',
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
    substractor = 3
    if(len(file_name_without_mime) == 8):
        substractor = 4
    expression_code = file_name_without_mime[len(
        file_name_without_mime)-substractor:len(file_name_without_mime)]
    return expression_code


for dirpath, dirnames, files in os.walk(KDEF_original_path):
    for file_name in files:
        expression_code = get_expression_code(file_name)
        copy_image(dirpath+'/'+file_name, expression_code, KDEF_filtered_path)
