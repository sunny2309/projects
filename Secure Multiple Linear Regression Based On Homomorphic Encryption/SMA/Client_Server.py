import random
import socket
import sys
import pickle
#from cPickle import dumps, loads
def openIngressSocket(myAddr, myPort):                          #open port for receiving data
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverAddress = (myAddr, myPort)

    sock.bind(serverAddress)
    sock.listen(1)

    connection, clientAddress = sock.accept()
    #print "clientAddress", clientAddress
    sock.close()
    return connection


def openEgressSocket(toAddr, toPort):                            #open port for sending data
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    toAddress = (toAddr, toPort)
    sock.connect(toAddress)

    return sock


def sendData(sock, data):                                         #Sending data (clients)
    try:
        sock.sendall(pickle.dumps(data))

    finally:
        return


def recvData(connection):                                       #Receiving data (clients)
    # initially make it block
    #print "Waiting for data"
    connection.settimeout(None)

    try:
        data = b''
        while True:
            data_new = connection.recv(1000)
            if data_new:
                data += data_new
                connection.settimeout(1)
            else:
                print ("Data null")
                break
    finally:
        return pickle.loads(data)


def connectToServer(serverAddr, serverPort):                    #Connecting Server to clients (clients)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverAddress = (serverAddr, serverPort)
    sock.connect(serverAddress)
    return sock


def waitServerData(serverConn):                                 # receiving data from server(clients)
    #print "Wait for server data"
    serverConn.settimeout(None)
    try:
        data = b''
        while True:
            data_new = serverConn.recv(1000)
            if data_new:
                data += data_new
                serverConn.settimeout(0.5)
            else:
                break
    finally:
        return pickle.loads(data)
def sendServerResponse(connection, response):                #sending data from server (server)
    #print "Sending server resp"
    try:
        connection.sendall(pickle.dumps(response))
    finally:
        return
def createServer(myAddr, myPort, numClients, pub):          #Creating connections to 3 clients (server)
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connections = []
    clientAddrs = []

    server_address = (myAddr, myPort)
    sock.bind(server_address)

    # Listen for incoming connections
    sock.listen(2)

    for i in range(numClients):
        # Wait for a connection
        connection, clientAddress = sock.accept()
        connections.append(connection)
        clientAddrs.append(clientAddress)

	#Send pub key
        sendServerResponse(connection, pub)

    return connections

def disconnectFromServer(sock):                 #closing the connections
    sock.close()