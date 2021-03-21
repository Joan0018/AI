from flask import render_template, request, Response
import os
from PIL import Image
from app.utils import pipeline_model, mobileNetImageDetection, mobileRealTimeDetection, videoCapture, genderRealTimeDetection, mobileNetVideoDetection

UPLOAD_FOLDER = 'static/uploads'


def base():
    return render_template('base.html')


def index():
    return render_template('index.html')


def genderClassification():
    return render_template('gender.html')


def getwidth(path):
    img = Image.open(path)
    size = img.size  # width and height
    aspect = size[0] / size[1]  # width / height
    w = 300 * aspect
    return int(w)


def openRealTime():
    return render_template('realTimeDetection.html')


def gender_Real():
    return Response(genderRealTimeDetection(), mimetype='multipart/x-mixed-replace; boundary=frame')


def genderRealTime():
    return render_template('realTimeGender.html')


def gender():
    if request.method == "POST":
        f = request.files['image']
        filename = f.filename
        path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(path)
        w = getwidth(path)

        # prediction (pass to pipeline model)
        pipeline_model(path, filename, color='bgr')

        return render_template('gender.html', fileupload=True, img_name=filename, w=w)

    return render_template('gender.html', fileupload=False, img_name="gender.png")


def gender_Video():
    if request.method == "POST":
        f = request.files['video']
        filename = f.filename
        path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(path)

        # prediction (pass to pipeline model)
        videoCapture(path, color='bgr')

        return render_template('genderVideo.html', fileupload=True)

    return render_template('genderVideo.html', fileupload=False, img_name="gender.png")


def objectDetection():
    if request.method == "POST":
        if request.form['submit_button'] == "upload & predict":
            f = request.files['image']
            filename = f.filename
            path = os.path.join(UPLOAD_FOLDER, filename)
            f.save(path)
            w = getwidth(path)

            # get the extension of image
            imgExtension = filename.split('.')[1]

            # call mobilenet
            detectedObject = mobileNetImageDetection(path, filename) + '.' + imgExtension
            return render_template('objectDetection.html', fileupload=True, img_name=filename, detected_name=detectedObject, w=w)
        else:
            # realTimeObject()
            openRealTime()

    return render_template('objectDetection.html', fileupload=False, img_name="objectDetection.png")


def object_Video():
    if request.method == "POST":
        f = request.files['video']
        filename = f.filename
        path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(path)

        # mobileNetDetection(path, 'Video')
        mobileNetVideoDetection(path)

        return render_template('objectDetectionVideo.html', fileupload=True)

    return render_template('objectDetectionVideo.html', fileupload=False)


def realTimeObject():
    # call mobilenet
    return Response(mobileRealTimeDetection(), mimetype='multipart/x-mixed-replace; boundary=frame')
