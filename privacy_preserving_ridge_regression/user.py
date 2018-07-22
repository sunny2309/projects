## user.py
import socket
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import math
import phe
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler
from phe import paillier
import bitstring
from bitstring import Bits
from secrets import token_hex
import copy
from Crypto.Cipher import DES
import pickle
import time

class User(object):
    def __init__(self,X_train,y_train):
        self.y_train = y_train
        self.X_train = np.around(X_train,2)
    
    def paillier_encrypt(self, A_i, b_i, precision=2):
        A_i_enc = np.zeros(A_i.shape,dtype=np.object)
        for i in range(A_i.shape[0]):
            for j in range(A_i.shape[1]):
                #print(A_i[i][j])
                A_i_enc[i][j] = self.public_key.encrypt(A_i[i][j],precision=precision)
                
        b_i_enc = np.zeros(b_i.shape,dtype=np.object)
        for i in range(b_i.shape[0]):
            b_i_enc[i] = self.public_key.encrypt(b_i[i],precision=precision)
        
        return A_i_enc, b_i_enc
    
    def calculate(self):
        A_i, b_i = np.zeros((self.X_train.shape[1],self.X_train.shape[1]),dtype=np.float32), np.zeros(self.X_train.shape[1],dtype=np.float32)
        for x_i,y_i in zip(self.X_train,self.y_train):
            A_i += np.dot(x_i.reshape(len(x_i), 1), x_i.reshape(len(x_i), 1).T)
            b_i += (x_i * y_i)
        A_i_enc,b_i_enc = self.paillier_encrypt(A_i, b_i, precision=2)
        return A_i_enc, b_i_enc

if __name__ == '__main__':

    df = pd.read_csv('auto-mpg.csv')
    df = df[df['horsepower'] != '?']
    cols = df.columns.tolist()
    cols.remove('mpg')
    cols.remove('car name')
    X = df[cols].values.astype(np.float32)
    y = df['mpg'].values.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=.85,test_size=.15)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    fourth = int(len(X_train)/ 4)
    print(fourth)
    user1 = User(X_train[:fourth],X_test[:fourth])
    user2 = User(X_train[fourth:2*fourth],X_test[fourth:2*fourth])
    user3 = User(X_train[2*fourth:3*fourth],X_test[2*fourth:3*fourth])
    user4 = User(X_train[3*fourth:],X_test[3*fourth:])
    s = socket.socket()
    #host = socket.gethostname()
    host = 'localhost'
    port = 12345
    s.connect((host, port))
    s.send(pickle.dumps('User: Please Send Public Key'))
    key = pickle.loads(s.recv(4096))
    while isinstance(key,str):
	    time.sleep(10)
	    s.close()
	    s = socket.socket()
	    s.connect((host, port))
	    s.send(pickle.dumps('User: Please Send Public Key'))
	    key = pickle.loads(s.recv(1024))
	    
    print(key)
    user1.public_key = key
    user2.public_key = key
    user3.public_key = key
    user4.public_key = key
    for user in [user1, user2, user3, user4]:
        s.close()
        s = socket.socket()
        s.connect((host, port))
        s.send(pickle.dumps('User: Sending Encrypted Messages'))
        s.send(pickle.dumps(user.calculate()))
    #print s.recv(1024)
    s.close()
