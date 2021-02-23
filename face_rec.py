import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime

def listdir_no_hidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list

def save_rec(name, distance):
    with open('recognized.csv', 'r+') as f:
        data = f.readlines()
        names_list = []
        for line in data:
            entry = line.split(',')
            names_list.append(entry[0])
        if name not in names_list:
            now = datetime.now()
            date_str = now.strftime('%d-%m-%Y')
            time_str = now.strftime('%H:%M:%S')
            FMT = '%H:%M:%S'
            time_delta = datetime.strptime(time_str,FMT) - datetime.strptime(time_str, FMT)
            f.writelines(f'\n{name}, {distance}, {date_str}, {time_str}')

images_path = '/Users/piotrsularz/Desktop/faceApp/images'

images = []
names = []

images_list = listdir_no_hidden(images_path)

for img in images_list:
    image = cv2.imread(f'{images_path}/{img}')
    images.append(image)
    names.append(os.path.splitext(img)[0])

print(f'Faces in memory: {names}')

encode_list_known = find_encodings(images)
print(f'Encoding completed. Number of faces: {len(encode_list_known)}')

video_capture = cv2.VideoCapture(0)

while True:
    ret, img = video_capture.read()
    img_scaled = cv2.resize(img, (0,0), None, 0.5, 0.5)
    img_scaled = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2RGB)

    faces_current_frame = face_recognition.face_locations(img_scaled)
    encode_current_frame = face_recognition.face_encodings(img_scaled, faces_current_frame)

    for encode_face, face_location in zip(encode_current_frame, faces_current_frame):
        matches = face_recognition.compare_faces(encode_list_known, encode_face)
        face_distance = face_recognition.face_distance(encode_list_known, encode_face)
        print(face_distance)
        match_index = np.argmin(face_distance)

        if matches[match_index]:
            name = names[match_index].upper()
            myface_distance = round(face_distance[match_index],2)
            print(myface_distance)
            print(name)
            save_rec(name, myface_distance)
            y1, x2, y2, x1 = face_location
            y1, x2, y2, x1 = 2*y1, 2*x2, 2*y2, 2*x1
            cv2.rectangle(img, (x1,y1),(x2,y2),(0,0,255),4)
            text = name + ' ' + str(myface_distance)
            cv2.putText(img, text, (x1, y1-15), cv2.FONT_HERSHEY_COMPLEX, 0.8,(0,0,255),1)

    win_name = "FACE RECOGNITION"
    cv2.namedWindow(win_name)
    cv2.imshow(win_name, img)
    cv2.waitKey(1)

cv2.destroyAllWindows()