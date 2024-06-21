import cv2 
import  os
import numpy as np
from PIL import Image

 
def generate_dataset():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    def face_extractor(img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        if faces is ():
            return None
        for(x,y,w,h) in faces:
            cropped_face = img[y:y+h,x:x+w]
        return cropped_face
    cap=cv2.VideoCapture(0)
    id=1
    count=0
    while True:
        ret,frame = cap.read()
        if face_extractor(frame) is not None:
            count+=1
            face = cv2.resize(face_extractor(frame),(200,200))
            face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            file_name_path = './data/user'+str(id)+ "."+str(count)+'.jpg'
            cv2.imwrite(file_name_path,face)
            cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow('Face Cropper',face)
        else:
            print("Face non détectée")
            pass
        if cv2.waitKey(1)==13 or count==100:
            break
    cap.release()
    cv2.destroyAllWindows()
def train_classifier(data_dir):
    path = [os.path.join(data_dir,f) for f in os.listdir(data_dir)]
    faces = []
    ids = []
    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img,'uint8')
        id = int(os.path.split(image)[1].split('.')[1])
        faces.append(imageNp)
        ids.append(id)
    ids = np.array(ids)
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces,ids)
    clf.write('haarcascade_eye.xml')
train_classifier('data')
    
def draw_boundary(img,classifier,scaleFactor,minNeighbors,color,text,clf):
    gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_image,scaleFactor,minNeighbors)
    for(x,y,w,h) in features:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,pred = clf.predict(gray_image[y:y+h,x:x+w])
        confidence = int(100*(1-pred/300))
        if confidence>75:
            if id==1:
                cv2.putText(img,"mohammed",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1,cv2.LINE_AA)
        else:
            cv2.putText(img,"Inconnu",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),1,cv2.LINE_AA)
    return img 
def recognize(img,clf,faceCascade):
    color = {"blue":(255,0,0),"red":(0,0,255),"green":(0,255,0),"white":(255,255,255)}
    img = draw_boundary(img,faceCascade,1.1,10,color['white'],"Face",clf)
    return img
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read('haarcascade_eye.xml')
video_capture = cv2.VideoCapture(0)
while True:
    ret,img = video_capture.read()
    img = recognize(img,clf,faceCascade)
    cv2.imshow("face detection",img)
    if cv2.waitKey(1)==13:
        break
video_capture.release()
cv2.destroyAllWindows()
