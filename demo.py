import argparse
import logging
import os

import cv2
import torch
import yt_dlp
from mivolo.data.data_reader import InputType, get_all_files, get_input_type
from mivolo.predictor import Predictor
from timm.utils import setup_default_logging

#### custom ####
import socket
import pickle
import struct
import requests
from playsound import playsound
import threading
import math
import sys
sys.path.append('/home/piai/GH/test/age_detection')

_logger = logging.getLogger("inference")


def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch MiVOLO Inference")
    return parser


def tts():
    #### TTS ####
    api_id = 'p6qb6ittqq'
    api_key = 'P7ZQfPF2yv4Rsl3qwcitDIbbyhFvvJAEfhY8ySkt'
    
    url = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"
    headers = {"X-NCP-APIGW-API-KEY-ID" : api_id,
                "X-NCP-APIGW-API-KEY" : api_key,
                "Content-Type" : "application/x-www-form-urlencoded"}
    
    text = "어서오세요. 메뉴를 주문하시겠습니까?"
    
    response = requests.post(url, headers=headers, data={"speaker" : "nara", "text" : text})
    return response

def play_welcome_sound(self, res):
    if res.status_code == 200:
        with open("0001.mp3", "wb") as f:
            f.write(res.content)
        playsound("0001.mp3")


def main():
    parser = get_parser()
    setup_default_logging()
    args = parser.parse_args()
    
    args.input = "./bus.jpg"
    args.output = ""
    args.detector_weights = "./checkpoints/yolov8x_person_face.pt"
    args.checkpoint = "./checkpoints/model_imdb_cross_person_4.22_99.46.pth.tar"
    args.device = "cuda:0"
    args.with_persons = "True"
    args.disable_faces = "False"
    args.draw = "False"

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    predictor = Predictor(args, verbose=True)

    #### socket #### 
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_ip = '141.223.140.61'
    port = 9999
    socket_address = (host_ip, port)
    print('HOST IP:', host_ip)

    server_socket.bind(socket_address)
    server_socket.listen(1)
    print("Listening at:", socket_address)
    
    data = b""
    payload_size = struct.calcsize("Q")
    client_socket, addr = server_socket.accept()
    print('Connection from:', addr)

    #### video #### - client로부터 영상 정보 받기
    while True:
        while len(data) < payload_size:
            data += client_socket.recv(4096)
            
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data) < msg_size:
            data += client_socket.recv(4096)
            
        frame_data = data[:msg_size]
        data = data[msg_size:]
        
        frame = pickle.loads(frame_data)
        detected_objects, out_im = predictor.recognize(frame)  # 객체 탐지, bounding box frame

        if detected_objects:    # 객체가 탐지됐다면
            first_detected_object = detected_objects[0] if isinstance(detected_objects, list) else detected_objects
            print(first_detected_object.ages)
            # age_data = math.floor(first_detected_object.ages) // 10
            # print(f"Age: {age_data}")

            #### TTS ####
            # response = tts()

            # sound_thread = threading.Thread(target=play_welcome_sound, args=(response,))
            # sound_thread.daemon = True
            # sound_thread.start()

            if out_im is not None:
                cv2.putText(out_im, 'Welcom', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            if out_im is not None:
                out_im = frame.copy()

        cv2.imshow("Client Video", out_im if out_im is not None else frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 

    #### video ####
    # cap = cv2.VideoCapture(0)
    # while True:
    #     ret, frame = cap.read()
    #     if ret:
    #         _, out_im = predictor.recognize(frame)
    #         cv2.imshow("a",out_im)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #     else:
    #         break
    
    #### image ###
    # img = cv2.imread(args.input)
    # _, out_im = predictor.recognize(img)

    # cv2.imshow("a",out_im)
    # cv2.waitKey(0)
    

if __name__ == "__main__":
    main()
