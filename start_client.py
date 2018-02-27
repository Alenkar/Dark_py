from ctypes import *
import asyncio
import sys, io
import struct
import time
import os
import cv2
import socket
import numpy
from PIL import Image, ImageDraw

start = time.time()

g = 100
image_path = "/home/alex/darknet_py_test/darknet/python/dog_2_1.jpg"

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('192.168.1.152', 31333))
print("CONNECT")

path_file = "/home/alex/Загрузки/Lada.mp4"
video_capture = cv2.VideoCapture(path_file)
x = 0
maxt = 0
mint = 1000
while 1:
    x += 1
#for x in range(g):
    try:
        start_time = time.time()
        #client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ret, frame = video_capture.read()
        if ret == 1:
            # connect the client
            # client.connect((target, port))
            #client.connect(('172.17.0.1', 30303))
            #print("CONNECT")

            s = 'Priem: ' + str(x)
            # send some data (in this case a HTTP GET request)

            cv2.imwrite(image_path, frame)
            with open(image_path, "rb") as imageFile:
                f = imageFile.read()
                b = bytearray(f)
            print("BYTEARR")

            #im_arr = numpy.fromstring(img.tobytes(), dtype=numpy.uint8)
            #im_arr = im_arr.reshape((img.size[1], img.size[0], 3))
            ln = sys.getsizeof(b)
            print(ln)
            #print(len(b))
            #im_ex = Image.open(io.BytesIO(b))
            #im_ex.show()
            ln_s = struct.pack('i', ln)
            print("SIZE")

            client.send(ln_s)
            #print(type(b))

            #client.send(b)
            print("Send")

            # receive the response data (4096 is recommended buffer size)
            #response = client.recv(4096)
            #print("MESS")

            #client.close()
            #print(response)

            end_time = time.time()
            fps = end_time - start_time

            if maxt <= fps:
                maxt = fps
            elif mint >= fps:
                mint = fps
        else:
            break
    except KeyboardInterrupt:
        print("INTERRUPT")
        client.close()
        break

client.close()
end = time.time()
life = end - start
print("Time: {0}".format(life))
print("Min={0} Max={1}".format(mint, maxt))
print("FPS = {0}".format(1/fps))