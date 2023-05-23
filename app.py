from flask import Flask, render_template, Response
import cv2
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
cascadePath = "Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
names = ['None','Dharmaraj','Vikram'] 
app=Flask(__name__)

def capture_by_frames(): 
    global cam
    cam = cv2.VideoCapture(0)
    while True:
        ret, img =cam.read()
        img = cv2.flip(img, 1) # 1 Stright 0 Reverse
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        detector=cv2.CascadeClassifier(cascadePath)
        faces=detector.detectMultiScale(img,1.2,6)
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])           
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "Unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            cv2.putText(img,str(id),(x+5,y-5),font,1,(255,255,255),2)
            #cv2.putText(img,str(confidence),(x+5,y+h),font,1,(255,255,0),1)
        ret1, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start',methods=['POST'])
def start():
    return render_template('index.html')

@app.route('/stop',methods=['POST'])
def stop():
    if cam.isOpened():
        cam.release()
    return render_template('stop.html')

@app.route('/video_capture')
def video_capture():
    return Response(capture_by_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True,use_reloader=False, port=8000)