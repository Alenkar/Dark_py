from ctypes import *
import os, io, time
from multiprocessing import Process
import threading
import cv2
import json
import asyncio
import sys
import struct
import numpy as np
from time import sleep
import datetime
import socket
import sys
import shutil
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def log_write(log_data):
    file = open('log_file.txt', 'a')
    if not "Average" in log_data and not "Load" in log_data:
        log_data += time.strftime(" Time=%x %H:%M:%S\n", time.localtime(time.time()))
    if "finished" in log_data:
        log_data += "\n\n"
    file.write(log_data)
    file.close()
    #print("Log создан")

logo = "------------------------------------------------------------------------------\n" \
       "|                INFORMATION LOG OF PROGRAM: NN YOLO/TINY                    |\n" \
       "------------------------------------------------------------------------------\n"

log_write(logo + "Start program")
#/home/alex/.ramdisk/
try:
    # Инициализация всех объектов
    print("Initialization all objects..")
    average_processed_image_time = 0.0
    max_processed_image_time = 0.0
    #error_name = "none"
    fps = 0
    #cv2.namedWindow('Video0', 0)
    #cv2.resizeWindow('Video0', 640, 480)
    # Назначение узла и порта, до которых будет стучаться клиент
    TCP_IP = '192.168.1.70'
    PORT = 31333
    im_serv_get = False
    exit_bool = False
    server_fall = False
    # Путь до файла изображения
    file_name = "/home/alex/PycharmProjects/DarknetYOLOdefault/dog.jpg"
    # Путь до файла данных (coco)
    data_name = b"/home/alex/darknet_py_test/darknet/cfg/coco.data"
    # Отладочные переменные для потоков
    iter_send = 0
    x = 1
    name_th = "Thread-1"
    name_th_start = "Thread-1"
    iter_i = 1
    # start_time = time.time()
    # Average, Min, Max PFS
    average_fps = 0.0
    min_fps = 1000
    max_fps = 0.0
    # Инициализация устройства захвата
    #path_file = "/home/alex/Загрузки/abrams.mp4"
    path_file = "/home/alex/Загрузки/Lada.mp4"
    # Захват с камеры или файла для инициализации изображения
    video_capture = cv2.VideoCapture(path_file)
    # Итерационная переменная для кадров
    frame_index = 0
    # Путь до изображения (позднее, путь до RAM-disk)
    #path = "/home/alex/darknet_py_test/darknet/python/alfa.jpg"
    path = "/home/alex/PycharmProjects/DarknetYOLOdefault/dog_1.jpg"
    b_path = path.encode()
    frame = cv2.imread(path)
    # Создание массива для обработки потоками (%3)
    frame_arr = [frame, frame, frame]

    # Время жизни программы
    s1 = time.time()
    # Итерационная переменная для потоков

    # Создание структуры для NN
    def c_array(ctype, values):
        arr = (ctype * len(values))()
        arr[:] = values
        return arr


    class BOX(Structure):
        _fields_ = [("x", c_float),
                    ("y", c_float),
                    ("w", c_float),
                    ("h", c_float)]


    class IMAGE(Structure):
        _fields_ = [("w", c_int),
                    ("h", c_int),
                    ("c", c_int),
                    ("data", POINTER(c_float))]


    class METADATA(Structure):
        _fields_ = [("classes", c_int),
                    ("names", POINTER(c_char_p))]


    # Путь до Си-шной библиотеки lib программы Darknet-YOLO
    lib = CDLL("/home/alex/darknet_py_test/darknet/libdarknet.so", RTLD_GLOBAL)
    # lib = CDLL("libdarknet.so", RTLD_GLOBAL)

    # Инициализация слоёв и объектов NN
    lib.network_width.argtypes = [c_void_p]
    lib.network_width.restype = c_int
    lib.network_height.argtypes = [c_void_p]
    lib.network_height.restype = c_int

    predict = lib.network_predict
    predict.argtypes = [c_void_p, POINTER(c_float)]
    predict.restype = POINTER(c_float)

    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

    make_image = lib.make_image
    make_image.argtypes = [c_int, c_int, c_int]
    make_image.restype = IMAGE

    make_boxes = lib.make_boxes
    make_boxes.argtypes = [c_void_p]
    make_boxes.restype = POINTER(BOX)

    free_ptrs = lib.free_ptrs
    free_ptrs.argtypes = [POINTER(c_void_p), c_int]

    num_boxes = lib.num_boxes
    num_boxes.argtypes = [c_void_p]
    num_boxes.restype = c_int

    make_probs = lib.make_probs
    make_probs.argtypes = [c_void_p]
    make_probs.restype = POINTER(POINTER(c_float))

    detect = lib.network_predict
    detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

    reset_rnn = lib.reset_rnn
    reset_rnn.argtypes = [c_void_p]

    load_net = lib.load_network
    load_net.argtypes = [c_char_p, c_char_p, c_int]
    load_net.restype = c_void_p

    free_image = lib.free_image
    free_image.argtypes = [IMAGE]

    letterbox_image = lib.letterbox_image
    letterbox_image.argtypes = [IMAGE, c_int, c_int]
    letterbox_image.restype = IMAGE

    load_meta = lib.get_metadata
    lib.get_metadata.argtypes = [c_char_p]
    lib.get_metadata.restype = METADATA

    load_image = lib.load_image_color
    load_image.argtypes = [c_char_p, c_int, c_int]
    load_image.restype = IMAGE

    rgbgr_image = lib.rgbgr_image
    rgbgr_image.argtypes = [IMAGE]

    predict_image = lib.network_predict_image
    predict_image.argtypes = [c_void_p, IMAGE]
    predict_image.restype = POINTER(c_float)

    network_detect = lib.network_detect
    network_detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

    print("Completed!")
    log_write("Initialization completed")
except Exception:#EnvironmentError:
    print("Error initialization")
    log_write("Error initialization")
    sys.exit(1)


'''
    ОТЛАДКА
'''

# Завершение программы и лог
def exit_program():
    global exit_bool, name_th
    exit_bool = True

    #log_programm()
    video_capture.release()
    cv2.destroyAllWindows()
    #sys.exit(1)

# Создание лога ошибки
def log_error(error_name):
    log_data = 'Error: {0}'.format(error_name)
    log_write(log_data)

# Лог при завершении программы
def log_programm():
    # Информация о работе программы
    log_status = "Status of program:\n"
    log_status += "Frame count: {1}\nTime of Life: {0}\n".format(round((time.time() - s1), 3), frame_index)
    try:
        log_status += "Average processed image time: {0}\n".format(round(average_processed_image_time / frame_index, 5))
        try:
            log_status += "Max processed image time: {0}\n".format(round(max_processed_image_time, 5))
            if min_fps == 1000:
                log_status += "FPS = 0 (Program Stack)\n"
            else:
                log_status += "FPS - Max: {0}, Min: {1}\n".format(round(max_fps, 3), round(min_fps, 3))
                log_status += "Average: {0}\n".format(round(1/(average_fps/frame_index), 3))
            log_write(log_status)
        except ZeroDivisionError:
            log_write(log_status)
            log_error("ZeroDivisionError in FPS counter")
    except ZeroDivisionError:
        log_write(log_status)
        log_error("ZeroDivisionError in processed image counter")

# Выбор запускаемой версии
arg = None
try:
    arg = sys.argv[1]
except:
    while 1:
        print("Введите сборку(t - tiny, y - yolo):")
        arg = input()
        if arg == 'y' or arg == 't':
            break
finally:
    if arg == 'y':
        # YoLO
        # Путь до файла конфигурации (config)
        cfg_name = b"/home/alex/darknet_py_test/darknet/cfg/yolo.cfg"
        # Путь до файла весов (weights)
        weight_name = b"/home/alex/darknet_py_test/darknet/yolo.weights"
    elif arg == 't':
        # Tiny-YoLO
        # Путь до конфигурации (config)
        cfg_name = b"/home/alex/darknet_py_test/darknet/cfg/tiny-yolo.cfg"
        # Путь до файла весов (weights)
        weight_name = b"/home/alex/darknet_py_test/darknet/tiny-yolo.weights"

log_write("Load:Cfg = {0}\nLoad:Weight = {1}\n".format(cfg_name, weight_name))

'''
count_server = 1

async def server_read(reader, writer):
    try:
        s = await reader.read(4096)
        print("String = {0}".format(s))
        global im_serv_get
        print('Writing in file...')
        if (im_serv_get == False):
            im_serv_get = True
            writer.write('1'.encode())

        rec_length = await reader.read(4096)

        length = struct.unpack('!i', rec_length)[0]
        print('Received length: ', length)

        buf = b''
        while sys.getsizeof(buf) < length:

            data = await reader.read(length)
            buf += data
            # print('data: ', data)
        global im_serv_get
        print('Writing in file...')
        if (im_serv_get == False):
            with open('%d.jpeg' % count_server, 'wb') as f:
                f.write(buf)
            im_serv_get = True
            writer.write('1'.encode())
        else:
            print("ПОка нет")

    except struct.error:
        print('Error')
        writer.write('1'.encode())


def server_init():

    loop = asyncio.get_event_loop()
    # coro = asyncio.start_server(server_read, '192.168.1.194', 30303, loop=loop)
    coro = asyncio.start_server(server_read, TCP_IP, PORT, loop=loop)
    print('Start server')
    server = loop.run_until_complete(coro)
    print('Serving on {}'.format(server.sockets[0].getsockname()))

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass

    server.close()
    loop.run_until_complete(server.wait_closed())
    loop.close()
'''



'''
    КЛИЕНТ
'''

'''
json_str_send = []
# Отправка json строки на сервер
def send_json():
    global name_th, name_th_start, iter_i, send_b
    while 1:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # connect the client
            # client.connect((target, port))
            client.connect(('127.0.0.1', 40404))
            while 1:
                try:
                    while len(json_str_send) != 0:
                        # send some encode data
                        client.sendall(json_str_send.pop().encode())
                        # receive the response data
                        response = client.recv(4096)
                        print(response)
                except():
                    log_error("Error encode data!")
        except ConnectionRefusedError:
            log_error("Not connections to server.")

task_send = threading.Thread(target=send_json, args=())
task_send.start()

'''

#b_rec = False
conn_b_s = False
json_str_global = '{' + '"function":"neyron","channel":1,"frame":0,"objects":[]' + '}'
json_str_global = json_str_global.replace("'", "")
json_arr = []
json_arr.append(json_str_global)

#json_str_global_true = json_str_global
# Отправка json строки на сервер


def send_json():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        while exit_bool != True:
            try:
                client.connect(('192.168.1.59', 30000))
                print("Connect to server")
                log_write("JSON Connect to server")
                while 1:
                    global conn_b_s#, b_rec
                    conn_b_s = True
                    '''
                        ОТЛАДКА
                    '''
                    try:
                        '''
                            ВРЕМЯ ИЗМЕНИТЬ
                        '''
                        time.sleep(0.05)

                        global json_str_global, json_arr
                        if len(json_arr) == 1:
                            str_json = json_arr[0]
                            #print(str_json)
                            client.send(str_json.encode())
                        elif len(json_arr) > 1:
                            #print("LEN TRUUE FIRST = {0}".format(len(json_arr)))
                            str_json = json_arr.pop(0)
                            print(str_json)
                            #print("LEN TRUUE SEC = {0}".format(len(json_arr)))
                            client.send(str_json.encode())

                        #json_str_global = json_str_global_true
                        #print(json_str_global_true)
                    except socket.error:
                        #client.close()
                        #print("ERROR JSON SEND")
                        log_error("SEND JSON encode data!")
                        break
            except socket.error:
                #log_error("Client SendJson not connect to server")
                client.close()
                log_write("CLIENT JSON close")
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except KeyboardInterrupt:
        client.close()
        log_write("EXIT JSON SEND")
'''
def send_json():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        while 1:
            try:
                client.connect(('192.168.1.59', 30000))
                print("Connect to server")
                global conn_b_s, b_rec
                conn_b_s = True
            except socket.error:
                client.close()
                pass
            else:
                while 1:
                    while b_rec == True: pass
                    try:
                        global json_str_global
                        client.send(json_str_global.encode())
                        json_str_global = json_str_global_true
                        #print(json_str_global_true)
                    except socket.error:
                        client.close()
                        print("ERROR JSON SEND")
                        #log_error("Error encode data!")
                        break

    except KeyboardInterrupt:
        client.close()
        log_error("Not connections to server.")
'''

'''
    ЛОГ ДОПИСАТЬ (?)
'''


# Извлечение информации и отрисовка объектов
def draw_detect(res, meta, boxes, i, probs, j):
    #class_obj = bytes.decode(meta.names[i], encoding='utf-8')
    #print("Class: {0}, Probs: {1}%".format(class_obj, round(probs[j][i] * 100.0, 1)))
    # Старые координаты


    x = int(boxes[j].x)
    y = int(boxes[j].y)
    w = int(boxes[j].w)
    h = int(boxes[j].h)
    # Позиции начальная (x1 и y1) и конечная (x2 и y2)

    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)

    # X и Y - центр, W - ширина, H - высота
    x_o = int(boxes[j].x)
    y_o = int(boxes[j].y)
    w_o = int(boxes[j].w)
    h_o = int(boxes[j].h)
    '''
    # Позиции начальная (x1 и y1) и конечная (x2 и y2)
    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)

    '''
    # Форматирование текста под удобный формат
    clazz = bytes.decode(meta.names[i], encoding='utf-8')
    # Создание Json строки с найденным объектом и добавление её в массив детектированных объектов
    js_obj = '{' + '"id": {0},"lostCount": 0,"clazz": "{1}","x": {2},"y": {3},"width": {4},"height": {5}' \
        .format(str(j), clazz, str(x1), str(y1), str(w_o), str(h_o)) + "}"
    res.append(js_obj)

# Создание Json строки из найденных объектов
def detect(net, meta, file_path, thresh=.5, hier_thresh=.5, nms=.45):
    # Загрузка изображения в network (True/False)
    bool_load_net = 1
    # Инициализация стартового времени детекции объекта на изображении
    time_rec = time.time()
    try:
        # Загрузка изображения в NN
        #shutil.copy(file_name, file_path)

        im = load_image(file_path.encode(), 0, 0)
        # Init array of special type to net

        # Func to Thread (?)
        boxes = make_boxes(net)
        probs = make_probs(net)
        num = num_boxes(net)

        # Детекция объектов с изображения
        network_detect(net, im, thresh, hier_thresh, nms, boxes, probs)

    except(FileExistsError, FileNotFoundError):
        #bool_load_net = 0
        log_error("Cannot load image")
        return None
    res = []
    if bool_load_net == 1:
        #print(meta.classes)
        for j in range(num):
            for i in range(meta.classes):
                if probs[j][i] > 0.25:
                    '''
                        Указать фильтр класса
                    '''
                    if i >= 0:
                        draw_detect(res, meta, boxes, i, probs, j)

    # Создание окончательной Json строки с массивом найденных объектов
    json_str = '{' + '"function": "neyron","channel": 1,"frame": {0},"objects": {1}'\
        .format(str(frame_index), str(res)) + '}'
    # Рассчёт времени обработки кадра
    processed_time = time.time() - time_rec
    global average_processed_image_time
    global max_processed_image_time
    average_processed_image_time += processed_time
    if max_processed_image_time < processed_time:
        max_processed_image_time = processed_time
    print("Processed image in: {0}\n".format(round(processed_time, 5)))
    # Очистка объектов из памяти
    free_image(im)
    free_ptrs(cast(probs, POINTER(c_void_p)), num)
    return json_str

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


# Функция запуска обработки нейросетью
def nn_funk(file_path):
    try:
        # Capture frame-by-frame
        # Время старта распознавания кадра
        start = time.time()
        ret = 1
        # Создание кадра с потока видео (True/False)
        if ret == 1:
            # Попытка приёма с сервера (?)
            json_str = detect(net, meta, file_path)
            json_str = json_str.replace("'", "")
            # Отправка json строки
            global name_th, iter_i
            name = threading.currentThread().getName()
            while 1:
                if name == name_th:
                    global json_str_global, json_arr#, b_rec
                    #b_rec = False
                    json_arr.append(json_str)
                    json_str_global = json_str

                    iter_i += 1
                    name_th = "Thread-" + str(iter_i)
                    print("Name new thread: {0} Old {1}".format(name_th, name))
                    break
            print("{0} is FINISH!".format(name))

        elif ret == 0:
            log_error("Cannot load frame")
        # Увеличение счётчика кадров
        global frame_index
        frame_index += 1
        # Конечное время распознавание кадра
        end = time.time()
        # Подсчёт времени на распознавание 1 кадра
        seconds = end - start
        # print("Time taken: {0} seconds".format(round(seconds, 5)))
        global average_fps
        average_fps += seconds
        # Вычисление кадров в секунду
        global fps
        fps = 1 / seconds
        # FPS на кадре
        global max_fps
        global min_fps
        if fps > max_fps:
            max_fps = fps
        elif fps < min_fps:
            min_fps = fps
        print("FPS : {0}".format(round(fps, 1)))

        # Выход из программы с помощью кнопки 'Q'
        '''
        ex_time_c = 60
        if fps > ex_time_c:
            ex_time = 1 / ex_time_c - 1 / (fps)
            sleep(ex_time)
        '''
        print("Time exec: {0}".format(round((1 / (time.time() - start)), 1)))

    except KeyboardInterrupt:
        log_error("Keyboard Interrupt Error")
        exit_program()

'''
    MAIN
'''

start_time = time.strftime("%x %H:%M:%S", time.localtime(time.time()))
set_gpu(0)
# Загрузка cfg_name, weight_name, data_name
net = load_net(cfg_name, weight_name, 0)
meta = load_meta(data_name)

iter_nn = 1
# Сервер

def server_init():
    global im_serv_get, iter_send, exit_bool
    sock_serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_serv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    while 1:
        try:
            sock_serv.bind((TCP_IP, PORT))
            log_write("Server BIND IP={0} PORT={1}".format(TCP_IP, PORT))
            break
        except socket.error:
            log_error("Bind Error")
            time.sleep(1)
    sock_serv.listen(1)

    '''
        ПРОВЕРИТЬ НА ОТЛАДКУ
    '''
    sock_serv.setblocking(0)

    print("Bind Server")
    log_write("Bind Server")

    task_json_send = threading.Thread(target=send_json, args=())
    task_json_send.start()
    while conn_b_s != True: pass
    log_write("Thread client start")
    while exit_bool != True:
        try:
            try:
                conn, addr = sock_serv.accept()
                log_write("Accept connect address={0}".format(addr))
                '''
                task = threading.Thread(target=send_json, args=())
                task.start()
                print("TH start")
                '''
                #while conn_b_s != True: pass
            except socket.error:
                pass
            else:
                while exit_bool != True:
                    #conn.setblocking(0)
                    try:
                        while 1:
                            #global b_rec
                            #print("Connection address:", addr)
                            # Загрузка размера изображения
                            #b_rec = True
                            try:
                                #print("111")
                                buf = conn.recv(4)
                                print("len buf = {0}".format(len(buf)))
                                print("buf = {0}".format(buf))
                                #print("222")
                            except socket.error:
                                print("Connect Refused")
                                log_error("Connect Refused (get Length)")
                                break
                            if len(buf) != 4:
                                print("LEN ERR")
                                ss = conn.recv(4096)
                                conn.send('1'.encode())
                                log_write("Connect Refused (len = {0})".format(len(buf)))
                                break
                            ln = struct.unpack('i', buf)[0]
                            #b_rec = False
                            #print("SIZE {0}".format(ln))
                            # Загрузка изображения
                            try:
                                size_im = ln
                                data = b''
                                while sys.getsizeof(data) < ln:
                                    data += conn.recv(size_im)
                                    size_im = ln - len(data)
                                print("SIZE GET {0}".format(sys.getsizeof(data)))

                                im_ex = Image.open(io.BytesIO(data))
                                file_path = "/home/alex/Test/{0}.jpg".format(iter_send)
                                im_ex.save(file_path)
                                '''
                                    THREAD ЗАПУСК ОБРАБОТЧИКА
                                '''
                                global name_th_start, iter_nn#, b_rec
                                #b_rec = True

                                print("Thread start")
                                name_th_start = "Thread-" + str(iter_nn)
                                task_send = threading.Thread(target=nn_funk, args=(file_path,), name=name_th_start)
                                task_send.start()
                                '''
                                    ОБНУЛЯТЬ ЛИ ITER_NN
                                '''

                                iter_nn += 1
                                while threading.active_count() > 10:
                                    pass
                                    #print("ACTIVE THREAD == {0}".format(threading.active_count()))

                                iter_send += 1

                                if iter_send == 10:
                                    iter_send = 0
                                    '''
                                    for thread in threading.enumerate():
                                        if thread.daemon:
                                            thread.join()
                                    '''
                                    #print("\n\nOBJECT NULL\n\n")
                                try:
                                    conn.send('1'.encode())
                                except socket.error:
                                    log_error("Connect Send ERROR")
                                    break
                                break
                            except struct.error:
                                print("Struct Error")
                                conn.send('1'.encode())
                                log_error("Structure is not true")
                    except socket.error:
                        log_write("Connect reactive")
                        break
                '''
                    ПРОВЕРКА ЗАКРЫТИЯ СОКЕТА (НЕ ЗАКРЫВАЛСЯ)
                '''
                conn.close()
        except KeyboardInterrupt:
            log_error("Keyboard Interrupt Error")
            exit_program()
    task_json_send.join()
    print("\nServer closed")
    sock_serv.shutdown(socket.SHUT_RDWR)
    sock_serv.close()
    log_write("Server closed")
    log_programm()
    log_write("Program is finished")


print("Init Server")
log_write("Init Server")
try:
    server_init()
except KeyboardInterrupt:
    log_error("KeyboardInterrupt")
    exit_program()
