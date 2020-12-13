import cv2
import dlib
import uuid
from scipy.spatial import distance
import time
import numpy as np
import math
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks
from keras.models import model_from_json
from keras.preprocessing import image
from pymongo import MongoClient
import json
import datetime
from bson import json_util
from nandishMongoClient import send_data 
myclient = MongoClient('mongodb+srv://[URL]')

mydb = myclient["new"]

mycol = mydb["test"]
count=0
data={}
data["Regid"]="RA1234"
data["uuid"]="False"
data['Yawn']="False"
data["Drowsy"]="False"
data["emotion"]="False"
data["headmove"]="False"
data["timestamp"]="False"

#load model
model = model_from_json(open("fer.json", "r").read())
#load weights
model.load_weights('fer.h5')


def mouth_aspect_ratio(lips):
    A = distance.euclidean(lips[2],lips[10])
    B = distance.euclidean(lips[4],lips[8])
    C = distance.euclidean(lips[0],lips[6])
    mar = (A+B)/(2.0*C)
    return mar
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B)/(2.0*C)
    return ear_aspect_ratio

def get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val):
    """Return the 3D points present as 2D for making annotation box"""
    point_3d = []
    dist_coeffs = np.zeros((4,1))
    rear_size = val[0]
    rear_depth = val[1]
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))
    
    front_size = val[2]
    front_depth = val[3]
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)
    
    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                    rotation_vector,
                                    translation_vector,
                                    camera_matrix,
                                    dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    return point_2d

def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix,
                        rear_size=300, rear_depth=0, front_size=500, front_depth=400,
                        color=(255, 255, 0), line_width=2):
    """
    Draw a 3D anotation box on the face for head pose estimation

    Parameters
    ----------
    img : np.unit8
        Original Image.
    rotation_vector : Array of float64
        Rotation Vector obtained from cv2.solvePnP
    translation_vector : Array of float64
        Translation Vector obtained from cv2.solvePnP
    camera_matrix : Array of float64
        The camera matrix
    rear_size : int, optional
        Size of rear box. The default is 300.
    rear_depth : int, optional
        The default is 0.
    front_size : int, optional
        Size of front box. The default is 500.
    front_depth : int, optional
        Front depth. The default is 400.
    color : tuple, optional
        The color with which to draw annotation box. The default is (255, 255, 0).
    line_width : int, optional
        line width of lines drawn. The default is 2.

    Returns
    -------
    None.

    """
    
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    # # Draw all the lines
    cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)
    
    
def head_pose_points(img, rotation_vector, translation_vector, camera_matrix):
    """
    Get the points to estimate head pose sideways    

    Parameters
    ----------
    img : np.unit8
        Original Image.
    rotation_vector : Array of float64
        Rotation Vector obtained from cv2.solvePnP
    translation_vector : Array of float64
        Translation Vector obtained from cv2.solvePnP
    camera_matrix : Array of float64
        The camera matrix

    Returns
    -------
    (x, y) : tuple
        Coordinates of line to estimate head pose

    """
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    y = (point_2d[5] + point_2d[8])//2
    x = point_2d[2]
    
    return (x, y)
        


cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat")

face_model = get_face_detector()
landmark_model = get_landmark_model()
#cap = cv2.VideoCapture(0)
ret, img = cap.read()
size = img.shape
font = cv2.FONT_HERSHEY_SIMPLEX 
# 3D model points.
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])

# Camera internals
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
                        [[focal_length, 0, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1]], dtype = "double"
                        )
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    faces_detected = face_haar_cascade.detectMultiScale(gray, 1.32, 5)
    # gray_img= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for (x,y,w,h) in faces_detected:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi_gray=gray[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        #find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]
        data["emotion"]=predicted_emotion
        cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    frame = cv2.resize(frame, (1000, 700))
   
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []
        lips=[]

        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n+1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            # cv2.line(frame, (x, y), (x2, y2), (255, 255, 255), 1)

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n+1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            # cv2.line(frame, (x, y), (x2, y2), (255, 255, 255), 1)
        for n in range(49,61):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            lips.append((x, y))
            next_point = n+1
            if n == 60:
                next_point = 49
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            # cv2.line(frame, (x, y), (x2, y2), (255, 255, 255), 1)


        left_ear =  calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)
        mouthaspect = mouth_aspect_ratio(lips)
        mouth_thresh=round(mouthaspect,2)
        EAR = (left_ear+right_ear)/2
        EAR = round(EAR, 2)

        if mouth_thresh>0.79:
            cv2.putText(frame, "Yawn", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 51, 51), 4)
            data["Yawn"]="True"
            # print("yawn")
            
        # print(mouth_thresh)                

        if EAR < 0.20:

            time.sleep(2)
            cv2.putText(frame, "DROWSY", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 51, 51), 4)
            cv2.putText(frame, "Are you Sleepy?", (50, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            data["Drowsy"]="True"
        #     print("Drowsy")
        # print(EAR)
        
        #serializedMyData = json.dumps(data, default=json_util.default)
        #x = mycol.insert_one(data)
        # send_data(data["Yawn"],data["Drowsy"])
        # print(data)
    if _ == True:                         
        faces = find_faces(frame, face_model)    
        for face in faces:
            marks = detect_marks(frame, landmark_model, face)
            # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
            image_points = np.array([
                                    marks[30],     # Nose tip
                                    marks[8],     # Chin
                                    marks[36],     # Left eye left corner
                                    marks[45],     # Right eye right corne
                                    marks[48],     # Left Mouth corner
                                    marks[54]      # Right mouth corner
                                ], dtype="double")
            dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
            
            
            # Project a 3D point (0, 0, 1000.0) onto the image plane.
            # We use this to draw a line sticking out of the nose
            
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            
            for p in image_points:
                cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
            
            
            p1 = ( int(image_points[0][0]), int(image_points[0][1]))
            p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            x1, x2 = head_pose_points(img, rotation_vector, translation_vector, camera_matrix)

            cv2.line(frame, p1, p2, (0, 255, 255), 2)            #img to frame
            cv2.line(frame, tuple(x1), tuple(x2), (255, 255, 0), 2)
            # for (x, y) in marks:
            #     cv2.circle(img, (x, y), 4, (255, 255, 0), -1)
            # cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
            try:
                m = (p2[1] - p1[1])/(p2[0] - p1[0])
                ang1 = int(math.degrees(math.atan(m)))
                # print("ang1:",ang1)
            except:
                ang1 = 90
                
            try:
                m = (x2[1] - x1[1])/(x2[0] - x1[0])
                ang2 = int(math.degrees(math.atan(-1/m)))
                # print("ang2:",ang2)
            except:
                ang2 = 90
                
                # print('div by zero error')
            if ang1 >= 20:
                data["headmove"]="Head down"
                # print('Head down')
                cv2.putText(frame, 'Head down', (30, 30), font, 2, (255, 255, 128), 3) 
            elif ang1 <= -48:
                data["headmove"]="Head up"
                # print('Head up')
                cv2.putText(frame, 'Head up', (30, 30), font, 2, (255, 255, 128), 3)
            
            if ang2 >= 25:
                data["headmove"]="Head right"
                # print('Head right')
                cv2.putText(frame, 'Head right', (90, 30), font, 2, (255, 255, 128), 3)
            elif ang2 <= -48:
                data["headmove"]="Head left"
                # print('Head left')
                cv2.putText(frame, 'Head left', (90, 30), font, 2, (255, 255, 128), 3)
              
                
            id=str(uuid.uuid1())
            # print(id)
            data["uuid"]=id
            data["timestamp"]=datetime.datetime.now()    
            send_data(data["Regid"],data["Yawn"],data["Drowsy"],data["emotion"],data["headmove"],data["uuid"],data["timestamp"])
            print(data)
            
               
            cv2.putText(frame, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
            cv2.putText(frame, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
#cv2.destroyAllWindows()
#cap.release()
    # serializedMyData = json.dumps(data, default=json_util.default)
    # x = mycol.insert_one(data)
    # print(x.inserted_id)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()