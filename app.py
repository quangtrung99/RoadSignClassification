from flask import Flask 
from flask import render_template, redirect, request, url_for, Response
from camera import VideoCamera

app = Flask(__name__)

def gen_frames(camera):  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        frame = camera.get_frame()  # read the camera frame
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/video')
def videoIndex():
    return render_template('video.html')

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=88, debug=True)