import os, io, time
import threading
import cv2
import sys
import struct
import numpy as np
from PIL import Image
import datetime
import socket
import matplotlib.pyplot as plt


cv2.namedWindow('video0', 0)
cv2.resizeWindow('video0', 640, 480)

TCP_IP = '127.0.0.1'
PORT = 30000
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind((TCP_IP, PORT))
    print("Init server")
except socket.error as exc:
    if exc.args[0] != 98:
        raise
        s.close()
        print("Error bind port")
        s.bind((TCP_IP, PORT))
s.listen(1)
iter_send = 0

#video = cv2.VideoCapture("/home/alex/Загрузки/Lada.mp4")


def funk():
    try:
        global addr_old, addr_arr
        while 1:
            #time.sleep(5)
            #ret, frame = video.read()
            conn, addr = s.accept()

            print("\nConnection address:", addr)
            data = conn.recv(4096)
            data_2 = data.decode('utf-8')
            print(data_2)
            data_3 = eval(data_2)
            objects_arr = data_3["objects"]


            '''
            if not data:
                print("ПЗДИЕЦ")
                break
            '''

            #img = cv2.imread("/home/alex/darknet_py_test/darknet/python/dog.jpg")
            # cv2.rectangle(img, )
            #cv2.imshow('OTR', img)
            global iter_send
            send_d = 'Json Get!! ' + str(iter_send)
            iter_send += 1
            #print("received data:", data)
            conn.send(send_d.encode())
            '''

            '''
            try:
                buf = conn.recv(4)
                if len(buf) != 4:
                    break
            except socket.error:
                print("Connect Refused")
                break
            ln = struct.unpack('i', buf)[0]

            # Загрузка изображения
            try:
                size_im = ln - 24
                data = b''
                while sys.getsizeof(data) <= (ln - 28):
                    data += conn.recv(size_im)
                    size_im = ln - sys.getsizeof(data)
                print(sys.getsizeof(data))
                print("CRETE IMAGE")
                # 1 метод
                try:
                    file_name = "/home/alex/PycharmProjects/DarknetYOLOdefault/img.jpg"
                    im_ex = Image.open(io.BytesIO(data))
                    im_ex.save(file_name)
                    img = cv2.imread(file_name)

                    # print(objects_arr)
                    for obj in objects_arr:
                        h = obj["height"]
                        w = obj["width"]
                        x = obj["x"]
                        y = obj["y"]
                        x1 = int(x - w / 2)
                        y1 = int(y - h / 2)
                        x2 = int(x + w / 2)
                        y2 = int(y + h / 2)
                        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    cv2.imshow('video0', img)
                except IOError:
                    print("err")
            except struct.error:
                print("struct error")

            conn.close()

            #print("COUNT CLIENT = {0}".format(io.sockets.clients().length()))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                conn.close()
                break
        #cv2.destroyAllWindows()
    except KeyboardInterrupt:
        print("Server closed")

funk()
