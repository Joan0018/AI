import time

import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import sklearn
import os
import pickle
import cv2
import face_recognition
import pathlib

OBJECT_IMAGE_EXPORT_PATH = 'static/object_detected/image/'
# Model for gender Classification
haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
# pickle files
mean = pickle.load(open('./model/mean_preprocess.pickle', 'rb'))
model_svm = pickle.load(open('./model/model_svm.pickle', 'rb'))
model_pca = pickle.load(open('./model/pca_50.pickle', 'rb'))

print('Model loaded sucessfully')

# settins
gender_pre = ['Male', 'Female']
font = cv2.FONT_HERSHEY_SIMPLEX


# Gender Classification (Image)
def pipeline_model(path, filename, color='bgr'):
    name = faceRecognitionImage(path)

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
        text = " %s : %0.2f" % (gender_pre[predict], score)
        txtName = "%s" % (name)
        cv2.putText(img, text, (x, y), font, 1, (255, 255, 0), 1)
        cv2.putText(img, txtName, (x, y + w), font, 1, (255, 255, 0), 1)

    cv2.imwrite('./static/predict/{}'.format(filename), img)


# Gender Classification (Video)
def videoCapture(path, color='bgr'):
    # step-1: read image in cv2
    cap = cv2.VideoCapture(path)

    while cap.isOpened:
        ret, frame = cap.read()

        if ret:
            frame = pipeline_model_video(frame, color='bgr')

            cv2.imshow('Gender Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):  # press s to exit  --#esc key (27),
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    return


def pipeline_model_video(img, color='rgb'):
    # step-2: convert into gray scale
    if color == 'bgr':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # step-3: crop the face (using haar cascase classifier)
    faces = haar.detectMultiScale(gray, 1.5, 3)
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # drawing rectangle
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
        cv2.putText(img, text, (x, y), font, 1, (255, 255, 0), 1)

    return img


def genderRealTimeDetection():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        frame = pipeline_model_video(frame, color='bgr', )

        img = cv2.imencode('.jpg', frame)[1].tobytes()
        yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n'


# Face Recognition
def faceRecognitionImage(path):
    # #loading the image to detect
    global name_of_person
    original_image = cv2.imread(path)

    # #load the sample images and get the 128 face embeddings from them
    modi_image = face_recognition.load_image_file('images/modi.jpg')
    modi_face_encodings = face_recognition.face_encodings(modi_image)[0]

    trump_image = face_recognition.load_image_file('images/trump.jpg')
    trump_face_encodings = face_recognition.face_encodings(trump_image)[0]

    joan_image = face_recognition.load_image_file('images/Joan.jpg')
    joan_face_encodings = face_recognition.face_encodings(joan_image)[0]

    lim_image = face_recognition.load_image_file('images/Lim.jpeg')
    lim_face_encodings = face_recognition.face_encodings(joan_image)[0]

    # #save the encodings and the corresponding labels in seperate arrays in the same order
    known_face_encodings = [modi_face_encodings, trump_face_encodings, lim_face_encodings, joan_face_encodings]
    known_face_names = ["Narendra Modi", "Donald Trump", "Lim Kah Yee", "Joan Hau"]

    # #load the unknown image to recognize faces in it
    image_to_recognize = face_recognition.load_image_file(path)

    # #detect all faces in the image
    # #arguments are image,no_of_times_to_upsample, model
    all_face_locations = face_recognition.face_locations(image_to_recognize, model='hog')
    # #detect face encodings for all the faces detected
    all_face_encodings = face_recognition.face_encodings(image_to_recognize, all_face_locations)

    # #print the number of faces detected
    print('There are {} no of faces in this image'.format(len(all_face_locations)))

    # #looping through the face locations and the face embeddings
    for current_face_location, current_face_encoding in zip(all_face_locations, all_face_encodings):
        #     #splitting the tuple to get the four position values of current face
        top_pos, right_pos, bottom_pos, left_pos = current_face_location

        #     #find all the matches and get the list of matches
        all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)

        #     #string to hold the label
        name_of_person = 'Unknown face'

        #     #check if the all_matches have at least one item
        #     #if yes, get the index number of face that is located in the first index of all_matches
        #     #get the name corresponding to the index number and save it in name_of_person
        if True in all_matches:
            first_match_index = all_matches.index(True)
            name_of_person = known_face_names[first_match_index]

    return name_of_person


def faceRecognitionVideo(path):
    # capture the video from default camera
    global left_pos, top_pos, right_pos, bottom_pos, name_of_person
    file_video_stream = cv2.VideoCapture(path)

    # load the sample images and get the 128 face embeddings from them
    # FACE #1
    aaron_Peirsol_image = face_recognition.load_image_file('images/samples/Aaron_Peirsol.jpg')
    aaron_Peirsol_face_encodings = face_recognition.face_encodings(aaron_Peirsol_image)[0]

    # FACE #2
    aaron_Sorkin_image = face_recognition.load_image_file('images/samples/Aaron_Sorkin.jpg')
    aaron_Sorkin_face_encodings = face_recognition.face_encodings(aaron_Sorkin_image)[0]

    # FACE #3
    abdel_Nasser_Assidi_image = face_recognition.load_image_file('images/samples/Abdel_Nasser_Assidi.jpg')
    abdel_Nasser_Assidi_face_encodings = face_recognition.face_encodings(abdel_Nasser_Assidi_image)[0]

    # FACE #4
    abdullah_image = face_recognition.load_image_file('images/samples/Abdullah.jpg')
    abdullah_face_encodings = face_recognition.face_encodings(abdullah_image)[0]

    # FACE #5
    trump_image = face_recognition.load_image_file('images/samples/trump.jpg')
    trump_face_encodings = face_recognition.face_encodings(trump_image)[0]

    # FACE #6
    abel_Pacheco_image = face_recognition.load_image_file('images/samples/Abel_Pacheco.jpg')
    abel_Pacheco_face_encodings = face_recognition.face_encodings(abel_Pacheco_image)[0]

    # FACE #7
    adam_Sandler_image = face_recognition.load_image_file('images/samples/Adam_Sandler.jpg')
    adam_Sandler_face_encodings = face_recognition.face_encodings(adam_Sandler_image)[0]

    # FACE #8
    adam_Scott_image = face_recognition.load_image_file('images/samples/Adam_Scott.jpg')
    adam_Scott_face_encodings = face_recognition.face_encodings(adam_Scott_image)[0]

    # FACE #9
    adolfo_Aguilar_Zinser_image = face_recognition.load_image_file('images/samples/Adolfo_Aguilar_Zinser.jpg')
    adolfo_Aguilar_Zinser_face_encodings = face_recognition.face_encodings(adolfo_Aguilar_Zinser_image)[0]

    # FACE #10
    ahmed_Chalabi_image = face_recognition.load_image_file('images/samples/Ahmed_Chalabi.jpg')
    ahmed_Chalabi_face_encodings = face_recognition.face_encodings(ahmed_Chalabi_image)[0]

    # FACE #11
    ai_Sugiyama_image = face_recognition.load_image_file('images/samples/Ai_Sugiyama.jpg')
    ai_Sugiyama_face_encodings = face_recognition.face_encodings(ai_Sugiyama_image)[0]

    # FACE #12
    aicha_El_Ouafi_image = face_recognition.load_image_file('images/samples/Aicha_El_Ouafi.jpg')
    aicha_El_Ouafi_face_encodings = face_recognition.face_encodings(aicha_El_Ouafi_image)[0]

    # FACE #13
    akbar_Hashemi_Rafsanjani_image = face_recognition.load_image_file('images/samples/Akbar_Hashemi_Rafsanjani.jpg')
    akbar_Hashemi_Rafsanjani_face_encodings = face_recognition.face_encodings(akbar_Hashemi_Rafsanjani_image)[0]

    # FACE #14
    akhmed_Zakayev_image = face_recognition.load_image_file('images/samples/Akhmed_Zakayev.jpg')
    akhmed_Zakayev_face_encodings = face_recognition.face_encodings(akhmed_Zakayev_image)[0]

    # FACE #15
    al_Gore_image = face_recognition.load_image_file('images/samples/Al_Gore.jpg')
    al_Gore_face_encodings = face_recognition.face_encodings(al_Gore_image)[0]

    # FACE #16
    alan_Ball_image = face_recognition.load_image_file('images/samples/Alan_Ball.jpg')
    alan_Ball_face_encodings = face_recognition.face_encodings(alan_Ball_image)[0]

    # FACE #17
    alberto_Fujimori_image = face_recognition.load_image_file('images/samples/Alberto_Fujimori.jpg')
    alberto_Fujimori_face_encodings = face_recognition.face_encodings(alberto_Fujimori_image)[0]

    # FACE #18
    alberto_Ruiz_Gallardon_image = face_recognition.load_image_file('images/samples/Alberto_Ruiz_Gallardon.jpg')
    alberto_Ruiz_Gallardon_face_encodings = face_recognition.face_encodings(alberto_Ruiz_Gallardon_image)[0]

    # FACE #19
    albrecht_Mentz_image = face_recognition.load_image_file('images/samples/Albrecht_Mentz.jpg')
    albrecht_Mentz_face_encodings = face_recognition.face_encodings(albrecht_Mentz_image)[0]

    # FACE #20
    alec_Baldwin_image = face_recognition.load_image_file('images/samples/Alec_Baldwin.jpg')
    alec_Baldwin_face_encodings = face_recognition.face_encodings(alec_Baldwin_image)[0]

    # FACE #21
    modi_image = face_recognition.load_image_file('images/modi.jpg')
    modi_face_encodings = face_recognition.face_encodings(modi_image)[0]

    # FACE #22
    joan_image = face_recognition.load_image_file('images/Joan.jpg')
    joan_face_encodings = face_recognition.face_encodings(joan_image)[0]

    # FACE #23
    lim_image = face_recognition.load_image_file('images/Lim.jpeg')
    lim_face_encodings = face_recognition.face_encodings(joan_image)[0]

    # save the encodings and the corresponding labels in seperate arrays in the same order
    known_face_encodings = [lim_face_encodings, joan_face_encodings, aaron_Peirsol_face_encodings,
                            aaron_Sorkin_face_encodings, abdel_Nasser_Assidi_face_encodings,
                            abdullah_face_encodings, trump_face_encodings,
                            abel_Pacheco_face_encodings, adam_Sandler_face_encodings, adam_Scott_face_encodings,
                            adolfo_Aguilar_Zinser_face_encodings, ahmed_Chalabi_face_encodings,
                            ai_Sugiyama_face_encodings,
                            aicha_El_Ouafi_face_encodings, akbar_Hashemi_Rafsanjani_face_encodings,
                            akhmed_Zakayev_face_encodings,
                            al_Gore_face_encodings, alan_Ball_face_encodings, alberto_Fujimori_face_encodings,
                            alberto_Ruiz_Gallardon_face_encodings, albrecht_Mentz_face_encodings,
                            alec_Baldwin_face_encodings,
                            modi_face_encodings]
    known_face_names = ["Lim Kah Yee", "Joan Hau", "Aaron Peirsol", "Aaron Sorkin", "Abdel Nasser Assidi",
                        "Abdullah", "Donald John Trump", "Abel Pacheco",
                        "Adam Sandler", "Adam Scott", "Adolfo Aguilar Zinser",
                        "Ahmed Chalabi", "Ai Sugiyama", "Aicha El Ouafi",
                        "Akbar Hashemi Rafsanjani", "Akhmed Zakayev", "Al Gore",
                        "Alan Ball", "Alberto Fujimori", "Alberto Ruiz Gallardon",
                        "Albrecht Mentz", "Alec Baldwin", "Narendra Modi"]

    # initialize the array variable to hold all face locations, encodings and names
    all_face_locations = []
    all_face_encodings = []
    all_face_names = []

    # loop through every frame in the video
    while file_video_stream.isOpened:
        # get the current frame from the video stream as an image
        ret, current_frame = file_video_stream.read()

        if ret:
            # resize the current frame to 1/4 size to proces faster
            current_frame_small = cv2.resize(current_frame, (0, 0), fx=0.25, fy=0.25)
            # detect all faces in the image
            # arguments are image,no_of_times_to_upsample, model
            all_face_locations = face_recognition.face_locations(current_frame_small, number_of_times_to_upsample=1,
                                                                 model='hog')

            # detect face encodings for all the faces detected
            all_face_encodings = face_recognition.face_encodings(current_frame_small, all_face_locations)

            # looping through the face locations and the face embeddings
            for current_face_location, current_face_encoding in zip(all_face_locations, all_face_encodings):
                # splitting the tuple to get the four position values of current face
                top_pos, right_pos, bottom_pos, left_pos = current_face_location

                # change the position maginitude to fit the actual size video frame
                top_pos = top_pos * 4
                right_pos = right_pos * 4
                bottom_pos = bottom_pos * 4
                left_pos = left_pos * 4

                # find all the matches and get the list of matches
                all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)

                # string to hold the label
                name_of_person = 'Unknown face'

                # check if the all_matches have at least one item
                # if yes, get the index number of face that is located in the first index of all_matches
                # get the name corresponding to the index number and save it in name_of_person
                if True in all_matches:
                    first_match_index = all_matches.index(True)
                    name_of_person = known_face_names[first_match_index]
                # draw rectangle around the face
            cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0, 255, 0), 2)

            # display the name as text in the image
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(current_frame, name_of_person, (left_pos, bottom_pos), font, 1, (255, 255, 0), 1)

            # display the video
            cv2.imshow("Webcam Video", current_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # release the stream and cam
    # close all opencv windows open
    file_video_stream.release()
    cv2.destroyAllWindows()


def faceRecognitionReal():
    # capture the video from default camera
    webcam_video_stream = cv2.VideoCapture(0)

    # load the sample images and get the 128 face embeddings from them
    modi_image = face_recognition.load_image_file('images/samples/modi.jpg')
    modi_face_encodings = face_recognition.face_encodings(modi_image)[0]

    trump_image = face_recognition.load_image_file('images/samples/trump.jpg')
    trump_face_encodings = face_recognition.face_encodings(trump_image)[0]

    joan_image = face_recognition.load_image_file('images/samples/Joan.jpg')
    joan_face_encodings = face_recognition.face_encodings(joan_image)[0]

    lim_image = face_recognition.load_image_file('images/Lim.jpeg')
    lim_face_encodings = face_recognition.face_encodings(lim_image)[0]

    # save the encodings and the corresponding labels in seperate arrays in the same order
    known_face_encodings = [modi_face_encodings, trump_face_encodings, lim_face_encodings, joan_face_encodings]
    known_face_names = ["Narendra Modi", "Donald Trump", "Lim Kah Yee", "Joan Hau"]

    # initialize the array variable to hold all face locations, encodings and names
    all_face_locations = []
    all_face_encodings = []
    all_face_names = []

    # loop through every frame in the video
    while True:
        # get the current frame from the video stream as an image
        ret, current_frame = webcam_video_stream.read()
        # resize the current frame to 1/4 size to proces faster
        current_frame_small = cv2.resize(current_frame, (0, 0), fx=0.25, fy=0.25)
        # detect all faces in the image
        # arguments are image,no_of_times_to_upsample, model
        all_face_locations = face_recognition.face_locations(current_frame_small, number_of_times_to_upsample=1,
                                                             model='hog')

        # detect face encodings for all the faces detected
        all_face_encodings = face_recognition.face_encodings(current_frame_small, all_face_locations)

        # looping through the face locations and the face embeddings
        for current_face_location, current_face_encoding in zip(all_face_locations, all_face_encodings):
            # splitting the tuple to get the four position values of current face
            top_pos, right_pos, bottom_pos, left_pos = current_face_location

            # change the position maginitude to fit the actual size video frame
            top_pos = top_pos * 4
            right_pos = right_pos * 4
            bottom_pos = bottom_pos * 4
            left_pos = left_pos * 4

            # find all the matches and get the list of matches
            all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)

            # string to hold the label
            name_of_person = 'Unknown face'

            # check if the all_matches have at least one item
            # if yes, get the index number of face that is located in the first index of all_matches
            # get the name corresponding to the index number and save it in name_of_person
            if True in all_matches:
                first_match_index = all_matches.index(True)
                name_of_person = known_face_names[first_match_index]

            # draw rectangle around the face
            cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (255, 0, 0), 2)

            # display the name as text in the image
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(current_frame, name_of_person, (left_pos, bottom_pos), font, 0.5, (255, 255, 255), 1)

        img = cv2.imencode('.jpg', current_frame)[1].tobytes()
        yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n'


# Object Detection
def mobileNetImageDetection(path, filename):
    # use for return the image name
    imgName = filename.split('.')[0]

    thres = 0.45  # Threshold to detect object

    classNames = []
    classFile = 'mobileNet/coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    net = initNet()

    img = cv2.imread(path)
    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):

            if confidence >= 0.5:

                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)

                if (box[2] - box[0]) < 400:
                    cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 120, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

                else:
                    cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

                imgName = imgName + '_' + classNames[classId - 1].capitalize() + '_' + str(
                    round(confidence * 100, 2)) + '%'

                exportImage = os.path.join(
                    os.path.join(pathlib.Path().parent.absolute(), OBJECT_IMAGE_EXPORT_PATH, imgName + '.jpg'))
                cv2.imwrite(exportImage, img)

    return imgName


def mobileRealTimeDetection():
    thres = 0.45  # Threshold to detect object

    cap = cv2.VideoCapture(0)

    classNames = []
    classFile = 'mobileNet/coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    net = initNet()

    # fps
    font = cv2.FONT_HERSHEY_PLAIN
    starting_time = time.time()
    frame_id = 0

    while True:
        success, img = cap.read()
        img = cv2.resize(img, (0, 0), fx=1.05, fy=1.0)

        frame_id += 1

        classIds, confs, bbox = net.detect(img, confThreshold=thres)

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                if confidence >= 0.5:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)

                    if (box[2] - box[0]) < 400:
                        cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

                        cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 120, box[1] + 30),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                    else:
                        cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

                        cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        cv2.putText(img, "FPS: " + str(round(fps, 2)), (10, 50), font, 2, (0, 0, 0), 3)

        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'


def mobileNetVideoDetection(path):
    thres = 0.45
    cap = cv2.VideoCapture(path)

    cap.set(3, 1280)
    cap.set(4, 720)
    cap.set(10, 70)

    classNames = []
    classFile = 'mobileNet/coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    net = initNet()

    while cap.isOpened:
        success, img = cap.read()

        if success:
            img = cv2.resize(img, (800, 600))

            classIds, confs, bbox = net.detect(img, confThreshold=thres)

            if len(classIds) != 0:
                for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):

                    if confidence >= 0.5:
                        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)

                        if (box[2] - box[0]) < 400:
                            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

                            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 120, box[1] + 30),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                        else:
                            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

                            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

            cv2.imshow("Output", img)
            if cv2.waitKey(1) & 0xFF == ord('s'):  # press s to exit  --#esc key (27),
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def initNet():
    configPath = 'mobileNet/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'mobileNet/frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    return net
