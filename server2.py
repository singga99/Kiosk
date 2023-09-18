from flask import Flask, request, render_template, Response, jsonify
import pickle
import cv2
from queue import Queue
from time import sleep

app = Flask(__name__)

frame_queue = Queue()

@app.route('/send_data', methods=['POST'])
def receive_data():
    data = request.data  # 클라이언트로부터 받은 데이터
    frame = pickle.loads(data)
    frame_queue.put(frame)  # 프레임을 큐에 추가
    cv2.imwrite('received_frame.jpg', frame)
    response = {'message' : 'Frame received and saved'}

    return jsonify(response), 200
    # return jsonify({"message" : "Frame received", "status" : "success"}), 200

def GenerateFrames():

    while True:
        if not frame_queue.empty():  # 큐가 비어있지 않을 경우
            frame = frame_queue.get()  # 큐에서 프레임을 가져옴
            
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def Index():
    return render_template('index.html')

@app.route('/stream')
def Stream():
    return Response(GenerateFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
