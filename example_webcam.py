from keras.models import load_model
import numpy as np 
import cv2
import time
import f_face_detector_occlusion
arg_input = "webcam"

# imagenes
list_img = ['0.jpg', '1.jpg', '2.jpg']
path_img = "data_test/"+list_img[2]

# instancio modelo de deteccion 
face_detector = f_face_detector_occlusion.detector_face_occlusion()


def bounding_box(img,box,match_name=[]):
    for i in np.arange(len(box)):
        x0,y0,x1,y1 = box[i].astype("int")
        img = cv2.rectangle(img,
                      (x0,y0),
                      (x1,y1),
                      (0,255,0),3);
        if not match_name:
            continue
        else:
            cv2.putText(img, match_name[i], (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    return img

#------------------------------------ FACE DETECTION -------------------------------------------
image = cv2.imread(path_img)
list_box = face_detector.detect_face(image)
# aplico bounding box
frame = bounding_box(image,list_box)
#------------------------------------ FACE RECOGNITION -------------------------------------------

# transform coord
list_box = list_box.astype("int")
# convierte coordenadas al siguiente formato(y0,x1,y1,x0)
#list(map(lambda x: (x[1],x[2],x[3],x[0]), list_box))
boxes = [(box[1],box[2],box[3],box[0]) for box in list_box]


import f_face_recognition as f  
f.detect_face(image)
f.get_features(image,boxes)
