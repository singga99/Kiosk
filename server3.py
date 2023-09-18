import argparse
import logging
import os

import cv2
import torch
import yt_dlp
from mivolo.data.data_reader import InputType, get_all_files, get_input_type
from mivolo.predictor import Predictor
from timm.utils import setup_default_logging

import socket
import struct
import pickle

_logger = logging.getLogger("inference")


def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch MiVOLO Inference")
    return parser


def main():
    parser = get_parser()
    setup_default_logging()
    args = parser.parse_args()
    
    args.input = "./dataset/messi.jpg"
    args.output = ""
    args.detector_weights = "checkpoints/yolov8x_person_face.pt"
    args.checkpoint = "checkpoints/model_imdb_cross_person_4.22_99.46.pth.tar"
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

    # data = b""
    # payload_size = struct.calcsize("Q")
    client_socket, addr = server_socket.accept()
    print('Connection from:', addr)

    #### pc video ####
    cap = cv2.VideoCapture(0)

    while True:
        recvData = client_socket.recv(4096)
        dist_data = float(recvData.decode('utf-8'))
        print(dist_data)

        ret, frame = cap.read()

        if ret:
            detected_objects, out_im = predictor.recognize(frame)

            if detected_objects:    # 객체가 탐지됐다면
                if detected_objects.ages:
                    first_detected_objects_ages = detected_objects.ages[0]
                    print(first_detected_objects_ages)
                    print(int(first_detected_objects_ages // 10))
                else:
                    print('No age info.')
            else:
                print('No objects detected.')

            if out_im is not None:
                cv2.imshow('Client Video', out_im)

            else:
                cv2.imshow('Client Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    #### video #### - client로부터 영상 정보 받기
    '''
    while True:
        recvData = client_socket.recv(4096)
        dist_data = float(recvData.decode('utf-8'))
        print(dist_data)
        
        
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
        detected_objects, out_im = predictor.recognize(frame)

        if detected_objects:    # 객체가 탐지됐다면
            if detected_objects.ages:
                first_detected_objects_ages = detected_objects.ages[0]
                print(first_detected_objects_ages)
                print(int(first_detected_objects_ages // 10))
            else:
                print('No age info.')
        else:
            print('No objects detected.')

        if out_im is not None:
            cv2.imshow('Client Video', out_im)

        else:
            cv2.imshow('Client Video', frame)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 
    '''

if __name__ == "__main__":
    main()
