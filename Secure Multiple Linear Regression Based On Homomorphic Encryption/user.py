import phe
from phe import paillier
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import sys
from Client_Server import *
import time

X_COLUMNS = ['cylinders','displacement','horsepower','weight','acceleration','model year','origin']
Y_COLUMN = 'mpg'
P = pow(2,128)
M = pow(2,64)

class User(object):
    
    def __init__(self,df, X_columns, Y_column,total_size,start,end, pub):
        X = df[X_columns].values.astype(np.float32)
        Y = df[Y_column].values.astype(np.float32)
        self.pub_key = pub
        
        Integer = np.vectorize(int)
        Float = np.vectorize(float)
        FixedPoint = np.vectorize(lambda x: x/ M if x <= pub.n else (x - pub.n) / M)
        
        print(X.dtype,Y.dtype)
        
        if start == 0:
            x_zeros = np.zeros((total_size-end, X.shape[1]))
            y_zeros = np.zeros(total_size - end)
            self.X = np.vstack((Integer(X*M), Integer(x_zeros*M)))
            self.Y = np.concatenate((Integer(Y*M), Integer(y_zeros*M)))
        elif end == total_size:
            x_zeros = np.zeros((start, X.shape[1]))
            y_zeros = np.zeros(start)
            self.X = np.vstack((Integer(x_zeros*M), Integer(X*M)))
            self.Y = np.concatenate((Integer(y_zeros*M), Integer(Y*M)))
        else:
            x_zeros1 = np.zeros((start-1,X.shape[1]))
            x_zeros2 = np.zeros((total_size-end,X.shape[1]))
            y_zeros1 = np.zeros((start-1,))
            y_zeros2 = np.zeros((total_size-end,))
            self.X = np.vstack((Integer(x_zeros1*M), Integer(X*M), Integer(x_zeros_2*M)))
            self.Y = np.concatenate((Integer(y_zeros1*M), Integer(Y*M), Integer(y_zeros2*M)))
        print(self.X.shape, self.Y.shape)
        
        self.XT = self.X.T
        self.X_enc = self.encrypt(self.X)
        self.X_enc_T = self.X_enc.T
        self.Y_enc = self.encrypt(self.Y)

    def encrypt(self,X):
        X_new = np.empty(X.shape,dtype=np.object)
        if len(X.shape) == 2:
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    #print(X[i][j])
                    X_new[i][j] = self.pub_key.encrypt(int(X[i][j]))
        else:
            for i in range(X.shape[0]):
                X_new[i] = self.pub_key.encrypt(int(X[i]))
                
        return X_new
    
    def decrypt(self,X,priv_key):  
        X_new = np.empty(X.shape,dtype=np.object)
        if len(X.shape) == 2:
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    X_new[i][j] = priv_key.decrypt(X[i][j])
        else:
            for i in range(X.shape[0]):
                X_new[i] = priv_key.decrypt(X[i])
        return X_new
    

    def calculate_parts(self,A1_enc, X_enc, X, XT,priv):
        Integer = np.vectorize(int)
        A2 = Integer(np.dot(XT,X))//M
        
        A_part1 = np.empty((A1_enc.shape[1],A1_enc.shape[1]),dtype=np.object)
        A_part2 = np.empty((A1_enc.shape[1],A1_enc.shape[1]),dtype=np.object)

        temp = (np.dot(X_enc.T, X) + np.dot(XT, X_enc))/M
        temp += A1_enc
        temp += A2
        t = self.decrypt(temp,priv)
        #print(t)
        print(t/M)
        for i in range(temp.shape[0]):
            for j in range(temp.shape[1]):
                r1 = random.randint(P, self.pub_key.n-1) % P
                #r1 = P+100
                A_part1[i][j] = temp[i][j] - r1
                A_part2[i][j] = r1
        
#        print(self.decrypt(A_part1,priv))
#        print(A_par2)
        return A_part1, A_part2  
                
    def calculate_c_inv(self,x1_enc,c1_enc, x2,c2):
        #print(x1_enc,c1_enc, x2, c2)
        pass
    
    @staticmethod
    def calculate_number_inverse(c):
        x = random.choice(np.linspace(0, 2/c, 11)[1:])
        for i in range(10):
            x *= (2 - c * x)
        return x
    
    @staticmethod
    def calculate_matrix_inverse(A):
        c = np.trace(A)
        I = np.eye(A.shape[0])
        #c_inv = calculate_number_inverse(c)
        c_inv = 1 / c
        print(c, c_inv)
        X0 = c_inv * I
        M0 = c_inv * A
        print(X0, M0)
        X = (2 * X0) - np.dot(X0, M0)
        M = (2 * M0) - np.dot(M0, M0)
        print(X, M)
        for i in range(1,50):
            X = (2 * X) - np.dot(X, M)
            M = (2 * M) - np.dot(M, M)
            #if np.allclose(M, np.eye(A.shape[0]), rtol = 0.001, atol = 0.001, equal_nan=True):
            #    break
        print(M, X)
        
        
if __name__ == '__main__':
    
    M, P = pow(2, 64), pow(2, 128)
    Integer = np.vectorize(int)
    Float = np.vectorize(float)
    FixedPoint = np.vectorize(lambda x: (x - P)/M if x > (P//2) else x/M)

    df = pd.read_csv('auto-mpg.csv')
    df = df[df.horsepower != '?']
    
    total_records = len(df)
    no_of_users = 2
    
    part_per_user = int(df.shape[0] / no_of_users)
    user_shares = []
    start = 0
    end = part_per_user
    for i in range(no_of_users):
        if i == no_of_users-1:
            end = df.shape[0]
            user_shares.append((start, end, df.iloc[start:,:]))
            continue
        user_shares.append((start,end, df.iloc[start:end, :]))
        start = end
        end += part_per_user
 
    
    user_id = int(sys.argv[1])
    if user_id == 0:
        conn =  openIngressSocket('localhost', 1234)
        pub, priv = paillier.generate_paillier_keypair(n_length = 1024)
        user = User(user_shares[user_id][2], X_COLUMNS,  Y_COLUMN,  total_records, user_shares[user_id][0], user_shares[user_id][1], pub)
        
        A = Integer(np.dot(user.XT,user.X))//M
        A_enc = user.encrypt(A)
        sendData(conn, {'X_enc':user.X_enc,'A1_enc': A_enc ,'pub_key': pub,'priv_key':priv})
        
        data = recvData(conn)
        A1 = user.decrypt(data['A1'],priv)
        #print(A1)
        A2 = data['A2']
        A1 = Integer(A1) % pub.n
        #print(A1)
        A1 = A1 % pub.n  
        A1 = (A1 - pub.n) % P
        #print(A1,A2)
        A = (A1+A2) % P
        
        print(FixedPoint(A))
        user.A = FixedPoint(A1)
        c1 = np.trace(user.A)
        c1 = int(c1 * M) % P
        x1 = random.choice(np.linspace(0, 2/c1, 11)[1:])
        x1 = 1e-12
        x1 = int(x1 * M) % P
        c1_enc = pub.encrypt(c1)
        x1_enc = pub.encrypt(x1)
        sendData(conn, {'c1_enc':c1_enc, 'x1_enc': x1_enc})
        #print(A1)
    else:
        sock = openEgressSocket('localhost', 1234)

        data = recvData(sock)
        print(data['pub_key'])
        pub, priv = data['pub_key'], data['priv_key']
        user = User(user_shares[user_id][2], X_COLUMNS,  Y_COLUMN,  total_records, user_shares[user_id][0], user_shares[user_id][1], data['pub_key'])
        
        A1, A2 = user.calculate_parts(data['A1_enc'],data['X_enc'], user.X, user.XT,data['priv_key'])
        FixedPoint = np.vectorize(lambda x: (x - P) if x > (P//2) else x)
        
        user.A = FixedPoint(A2)
        sendData(sock, {'A1':A1,'A2':A2})
        ## Calculate C_inv
        c2 = np.trace(user.A)
        c2 = int(c2*M)%P
        x2 = random.choice(np.linspace(0, 2/c2, 11)[1:])
        x2 = 1e-12
        x2 = int(x2*M)%P
        data = recvData(sock)
        print(priv.decrypt(data['x1_enc']),priv.decrypt(data['c1_enc']))
        user.calculate_c_inv(data['x1_enc'],data['c1_enc'], x2, c2)
