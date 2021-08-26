import cv2 as cv
from time import localtime, strftime
import os
from facenet_pytorch import MTCNN
import numpy as np
import os.path
from matplotlib.patches import Rectangle
from keras.models import load_model
from matplotlib.patches import Circle
import cv2
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import cosine
import os
import datetime
from os import path
from pathlib import Path
import pickle
import os.path
from flask import flash,url_for,redirect,render_template

encoder_model = 'facenet_keras.h5'
class Camera(object):
    CAPTURES_DIR = "static/captures/"
    RESIZE_RATIO = 1.0
    detector=MTCNN()
    face_encoder = load_model(encoder_model)
    l2_normalizer = Normalizer('l2')

    encoding_dict = dict()

    def start_cam(self):############ Selection of camera is needed
        # self.video = cv.VideoCapture('rtsp://admin:admin@123@192.168.1.240:554/cam/realmonitor?channel=2&subtype=0')
        self.video = cv.VideoCapture(0)
    
    def stop_cam(self):
        self.video.release()
        
    def get_face(self,img, box):
        [[x1, y1, width, height]] = box
        x1, y1 ,x2,y2= int(x1), int(y1),int(width),int(height)
        face = img[y1:y2, x1:x2]
        return face, (x1, y1), (x2, y2)

    
        # face = img[y1:y2, x1:x2]
        # return face, (x1, y1), (x2, y2)
    def normalize(self,img):
        mean, std = img.mean(), img.std()
        return (img - mean) / std

    def detect(self,frame):
        boxes,probs = Camera.detector.detect(frame)
        
        encodes=[]
        encode=[]

	       
        if boxes is not None:
            print(boxes)

            face, _, _ = self.get_face(frame, boxes)
            face = self.normalize(face)
            face = cv2.resize(face,(160,160))
            encode = Camera.face_encoder.predict(np.expand_dims(face, axis=0))[0]
            encodes.append(encode)
        if encodes:
            encode = np.sum(encodes, axis=0)
            encode = Camera.l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]


        return encode
    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None

        if (Camera.RESIZE_RATIO != 1):
            frame = cv.resize(frame, None, fx=Camera.RESIZE_RATIO, \
                fy=Camera.RESIZE_RATIO)    
        return frame

    def get_feed(self):
        frame = self.get_frame()
        frame=cv2.resize(face,(224,224))
        if frame is not None:
            ret, jpeg = cv.imencode('.jpg', frame)
            return jpeg.tobytes()

    def capture(self,name_f):
        frame = self.get_frame()
        timestamp = strftime("%d-%m-%Y-%Hh%Mm%Ss", localtime())
        
        #os.makedirs('photo/'+str(name_f))
        encode=[]
        encode=self.detect(frame)

        print("Encoder",encode)
        
        # if len(encode)==0:
        #     flash(1)
        #     return render_template("index.html")
        
        
        #print ("directory exists:" + str(path.exists('photo/' + str(name_f)+'_'+ str(name_l))))
        if not os.path.exists('static/photo/'):
            os.makedirs('static/photo/')
        filename ='static/photo/'+str(name_f)+ ".jpg"

        #encoded={}
        #filename = str(name_f)+'_'+ str(name_l)+'/'+timestamp +".jpg"
        #encoded={:encode}

        print("Filename for teacher",filename)

        if not cv.imwrite(filename, frame):
            raise RuntimeError("Unable to capture image "+timestamp)
        """with open('embeddings/'+str(name_f)+'_'+str(name_l)+'.dat', 'wb') as f:
            print('done')
            pickle.dump(encode, f)"""
        if os.path.exists('static/embeddings/'+str(name_f)+'.dat'):
            print("True")
            with open('static/embeddings/'+str(name_f)+'.dat',"rb") as f:
                encoded = pickle.load(f)
                #print(obj)



            with open('static/embeddings/'+str(name_f)+'.dat', 'wb') as f1:
            
                encoded[str(name_f)]=encode
                pickle.dump(encoded,f1)
        else:
            encoded={}
            with open('static/embeddings/'+str(name_f)+'.dat', 'wb') as f2:
            
                encoded[str(name_f)]=encode
                pickle.dump(encoded,f2)
            
        return timestamp,frame 
    """def capture_others(self,name_f,name_l):
        frame = self.get_frame()
        encode=self.detect(frame)
        print(encode)
        timestamp = strftime("%d-%m-%Y-%Hh%Mm%Ss", localtime())
        #os.mkdir(str(name_f)+"_"+str(name_l))
        #print ("directory exists:" + str(path.exists(Camera.CAPTURES_DIR + str(name_f)+'_'+ str(name_l))))
        

        filename ='photo/'+ str(name_f)+'_'+ str(name_l)+'/'+timestamp +".jpg"
        print(filename)
        with open('en.dat', 'wb') as f:
            print('done')
            pickle.dump(encode, f)
        if not cv.imwrite(filename, frame):
            raise RuntimeError("Unable to capture image "+timestamp)
        return timestamp,frame,filename"""