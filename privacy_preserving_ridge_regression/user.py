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
    def __init__(self):
        df = pd.read_csv('auto-mpg.csv')
        df = df[df['horsepower'] != '?']
        cols = df.columns.tolist()
        cols.remove('mpg')
        cols.remove('car name')
        self.X = df[cols].values.astype(np.float32)
        self.y = df['mpg'].values.astype(np.float32)
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,train_size=.8,test_size=.2)
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)
        self.X_train = np.around(self.X_train,2)
        self.X_test = np.around(self.X_test,2)
        #self.temp = np.around(np.dot(self.X_train.T,self.X_train),2)
        #print(np.around(np.dot(self.X_train.T,self.X_train),2))
    
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
        c = []
        #print(self.X_train.shape,self.y_train.shape)
        i = 1
        c_i = [ np.zeros((self.X_train.shape[1],self.X_train.shape[1])), np.zeros(self.X_train.shape[1])]
        for x_i,y_i in zip(self.X_train,self.y_train):
            #print(x_i.shape, y_i.shape,c_i[0].shape,c_i[1].shape)
            A_i = np.dot(x_i.reshape(len(x_i), 1), x_i.reshape(len(x_i), 1).T)
            b_i = x_i * y_i
            #print(A_i.shape, b_i.shape)
            if i%20 == 0:
                c_i_enc = self.paillier_encrypt(c_i[0], c_i[1], precision=2)
                c.append(c_i_enc)
                c_i = [ np.zeros((self.X_train.shape[1],self.X_train.shape[1])), np.zeros(self.X_train.shape[1])]
                print('Number of Records Encrypted %s. Currently Pending : %s'%(i, self.X_train.shape[0]-i))
            else:
                c_i[0] += A_i
                c_i[1] += b_i
            i += 1
        c_i_enc = self.paillier_encrypt(c_i[0], c_i[1], precision=2)
        c.append(c_i_enc)
        #c_mu = self.paillier_encrypt(self.mu_A, self.mu_b, precision=2)
        #self.c.append(c_mu)
        return c

if __name__ == '__main__':

    s = socket.socket()
    #host = socket.gethostname()
    host = 'localhost'
    port = 12345
    s.connect((host, port))
    s.send('User: Please Send Public Key')
	key = s.recv(1024).decode()
	
	while 'Please try again after sometime' in key:
	    time.sleep(10)
	    s.send('User: Please Send Public Key')
	    key = s.recv(1024).decode()
	    
    print(key)
    
    s.send('User: Sending Encrypted Messages')
    start = time.time()
    print('Calculating encrypted C and sending to Evaluator started')
    s.send(pickle.dumps(user.calculate()))
    print('Calculating encrypted Cs and sending to Evaluator completed.')
    print('Total time taken in calculating Cs : %s seconds'%(time.time()-start))
    #print s.recv(1024)
    s.close()
