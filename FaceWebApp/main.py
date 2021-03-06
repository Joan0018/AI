from flask import Flask
from app import views
app = Flask(__name__)

# URL
app.add_url_rule('/base','base',views.base)
app.add_url_rule('/','index',views.index)
app.add_url_rule('/genderClassification','gender',views.gender, methods=['GET','POST'])
app.add_url_rule('/faceRecognitionVideo','faceRecognitionVideo',views.faceRecognition_Video, methods=['GET','POST'])
app.add_url_rule('/faceRecognitionReal/openFaceRealTime','faceRecognitionReal',views.faceRealTime, methods=['GET','POST'])
app.add_url_rule('/faceRecognitionReal','realTimeFace',views.face_Real, methods=['GET','POST'])
app.add_url_rule('/objectDetection','objectDetection',views.objectDetection, methods=['GET','POST'])
app.add_url_rule('/realTimeObject','realTimeObject',views.realTimeObject, methods=['GET','POST'])
app.add_url_rule('/realTimeObject/openRealTime','openRealTime',views.openRealTime,  methods=['GET','POST'])
app.add_url_rule('/objectVideo','object_Video',views.object_Video,  methods=['GET','POST'])

# RUN
if __name__=="__main__":
    app.run(debug=True)