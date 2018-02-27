import socket

TCP_IP = '172.17.0.1'
PORT = 31333
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((TCP_IP, PORT))
print("Init server")
s.listen(1)
try:
    while 1:
        conn, addr = s.accept()
        print("\nСоединение от:", addr)
        iter = conn.recv(4096)
        print("ПРИЁМ: {0}".format(iter.decode('utf-8')))
        str = "Привет {0} друг! Ты лучший!".format(iter.decode('utf-8'))
        conn.send(str.encode())
        iter = conn.recv(4096)
        print("ПРИЁМ: {0}".format(iter.decode('utf-8')))
        conn.close()

except KeyboardInterrupt:
    print("Server closed")
