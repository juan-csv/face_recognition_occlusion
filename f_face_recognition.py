import face_recognition 
import numpy as np
import config as cfg
import f_face_detector_occlusion


face_detector = f_face_detector_occlusion.detector_face_occlusion()
def detect_face(image):
    '''
    Input: imagen numpy.ndarray, shape=(W,H,3)
    Output: [(y0,x1,y1,x0),(y0,x1,y1,x0),...,(y0,x1,y1,x0)] ,cada tupla representa un rostro detectado
    si no se detecta nada  --> Output: []

    antes -->    box_faces = face_recognition.face_locations(image)
    '''
    list_box = face_detector.detect_face(image)
    try:
        box_faces = [(box[1],box[2],box[3],box[0]) for box in list_box.astype("int")]
    except:
        box_faces = []
    return box_faces


def get_features(img,box):
    '''
    Input:
        -img:imagen numpy.ndarray, shape=(W,H,3)
        -box: [(y0,x1,y1,x0),(y0,x1,y1,x0),...,(y0,x1,y1,x0)] ,cada tupla representa un rostro detectado
    Output:
        -features: [array,array,...,array] , cada array representa las caracteristicas de un rostro 
    '''
    features = face_recognition.face_encodings(img,box)
    return features

def compare_faces(face_encodings,db_features,db_names):
    '''
    Input:
        db_features = [array,array,...,array] , cada array representa las caracteristicas de un rostro 
        db_names =  array(array,array,...,array) cada array representa las caracteriticas de un usuario
    Output:
        -match_name: ['name', 'unknow'] lista con los nombres que hizo match
        si no hace match pero hay una persona devuelve 'unknow'
    '''
    match_name = []
    names_temp = db_names
    Feats_temp = db_features           

    for face_encoding in face_encodings:
        try:
            dist = face_recognition.face_distance(Feats_temp,face_encoding)
        except:
            dist = face_recognition.face_distance([Feats_temp],face_encoding)
        index = np.argmin(dist)
        if dist[index] <= cfg.threshold:
            match_name = match_name + [names_temp[index]]
        else:
            match_name = match_name + ["unknow"]
    return match_name