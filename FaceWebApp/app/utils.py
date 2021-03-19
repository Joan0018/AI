import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import sklearn
import pickle
import cv2

#Model for gender Classification
haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
# pickle files
mean = pickle.load(open('./model/mean_preprocess.pickle', 'rb'))
model_svm = pickle.load(open('./model/model_svm.pickle', 'rb'))
model_pca = pickle.load(open('./model/pca_50.pickle', 'rb'))

print('Model loaded sucessfully')

# settins
gender_pre = ['Male', 'Female']
font = cv2.FONT_HERSHEY_SIMPLEX

#Gender Classification (Image)
def pipeline_model(path, filename, color='bgr'):
    # step-1: read image in cv2
    img = cv2.imread(path)

    # step-2: convert into gray scale
    if color == 'bgr':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # step-3: crop the face (using haar cascase classifier)
    faces = haar.detectMultiScale(gray, 1.5, 3)
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)  # drawing rectangle
        roi = gray[y:y + h, x:x + w]  # crop image

        # step - 4: normalization (0-1)
        roi = roi / 255.0

        # step-5: resize images (100,100)
        if roi.shape[1] > 100:
            roi_resize = cv2.resize(roi, (100, 100), cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi, (100, 100), cv2.INTER_CUBIC)

        # step-6: Flattening (1x10000)
        roi_reshape = roi_resize.reshape(1, 10000)  # 1,-1

        # step-7: subptract with mean
        roi_mean = roi_reshape - mean

        # step -8: get eigen image
        eigen_image = model_pca.transform(roi_mean)

        # step -9: pass to ml model (svm)
        results = model_svm.predict_proba(eigen_image)[0]

        # step -10:
        predict = results.argmax()  # 0 or 1
        score = results[predict]

        # step -11:
        text = "%s : %0.2f" % (gender_pre[predict], score)
        cv2.putText(img, text, (x, y), font, 1, (255, 255, 0), 2)

    cv2.imwrite('./static/predict/{}'.format(filename), img)

#Gender Classification (Video)
def videoCapture(path, color='bgr'):
    # step-1: read image in cv2
    cap = cv2.VideoCapture(path)

    while True:
        ret, frame = cap.read()

        frame = pipeline_model_video(frame,color='bgr',)

        cv2.imshow('Gender Detector',frame)
        if cv2.waitKey(10) == ord('s'): # press s to exit  --#esc key (27), 
          break
        
    cv2.destroyAllWindows()
    cap.release()

def pipeline_model_video(img,color='rgb'):
    # step-2: convert into gray scale
    if color == 'bgr':
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # step-3: crop the face (using haar cascase classifier)
    faces = haar.detectMultiScale(gray,1.5,3)
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) # drawing rectangle
        roi = gray[y:y+h,x:x+w] # crop image
        # step - 4: normalization (0-1)
        roi = roi / 255.0
        # step-5: resize images (100,100)
        if roi.shape[1] > 100:
            roi_resize = cv2.resize(roi,(100,100),cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi,(100,100),cv2.INTER_CUBIC)
        # step-6: Flattening (1x10000)
        roi_reshape = roi_resize.reshape(1,10000) # 1,-1
        # step-7: subptract with mean
        roi_mean = roi_reshape - mean
        # step -8: get eigen image
        eigen_image = model_pca.transform(roi_mean)
        # step -9: pass to ml model (svm)
        results = model_svm.predict_proba(eigen_image)[0]
        # step -10:
        predict = results.argmax() # 0 or 1 
        score = results[predict]
        # step -11:
        text = "%s : %0.2f"%(gender_pre[predict],score)
        cv2.putText(img,text,(x,y),font,1,(255,255,0),2)
    return img

def genderRealTimeDetection():

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        frame = pipeline_model_video(frame,color='bgr',)

        cv2.imshow('Gender Detector',frame)
        if cv2.waitKey(10) == ord('s'): # press s to exit  --#esc key (27), 
          break
        
    cv2.destroyAllWindows()
    cap.release()


#Object Detection
def mobileNetImageDetection(path, filename):
    thres = 0.45  # Threshold to detect object

    classNames = []
    classFile = 'mobileNet/coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    configPath = 'mobileNet/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'mobileNet/frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    img = cv2.imread(path)

    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    # print(classIds,bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imwrite('./static/predict/{}'.format(filename), img)


def mobileNetRealTimeDetection():
    thres = 0.45  # Threshold to detect object

    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)

    classNames = []
    classFile = 'mobileNet/coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    configPath = 'mobileNet/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'mobileNet/frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    while True:
        success, img = cap.read()
        img = cv2.resize(img, (0, 0), fx=0.6, fy=0.6)

        classIds, confs, bbox = net.detect(img, confThreshold=thres)

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
