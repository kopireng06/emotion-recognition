import cv2
import matplotlib.pyplot as plt
from PIL import Image

cropped_image_path = 'KDEF_CROPPED_test'
size = 48, 48

def crop_image(path, expression_code) :
    
    cascPath = "haarcascade_frontalface_default.xml"
    
    faceCascade = cv2.CascadeClassifier(cascPath)

    image = cv2.imread(path)
    image_crop = Image.open(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=2,
        minSize=(40, 60),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    im_crop = image_crop.crop((x, y, (x+w), (y+h))).convert('L')
    im_crop.thumbnail(size, Image.ANTIALIAS)
    
    namefile = path.split('/')[1] +'/'+ path.split('/')[2]
    im_crop.save(cropped_image_path + '/' + namefile)


import os
 
directory = 'KDEF_test/disgust'
abs_path = os.path.abspath("")
cropped_directories = ['angry', 'fear', 'disgust', 'happy', 'neutral', 'sad', 'surprised']
 
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

def get_expression_code(file_name):
    file_name_without_mime = file_name.split('.')[0]
    substractor = 3
    if(len(file_name_without_mime) == 8) :
        substractor = 4
    expression_code = file_name_without_mime[len(
        file_name_without_mime) - substractor : len(file_name_without_mime)]
    return expression_code

for dir in cropped_directories:
    try:
        os.makedirs(abs_path+'/'+cropped_image_path+'/' + dir)
    except:
        print("Directory Already Exists!")

for root, dirs, files in os.walk(directory):
    for filename in files:
        crop_image(os.path.join(root, filename), get_expression_code(filename))