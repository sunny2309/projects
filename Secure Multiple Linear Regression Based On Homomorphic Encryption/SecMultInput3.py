from random import *
from pandas import *
import pandas as pd
import random
from random import choice, randint
import numpy as np
from numpy.linalg import inv
from sympy.polys.galoistools import *
from sympy import *
import gmpy2
from gmpy2 import mpz, mpfr, log2
import csv
from sklearn.model_selection import train_test_split
a = 12345678901234567890
gmpy2.get_context().precision = 70
mpz(2 ** log2(a))
mpz(12345678901234567890)

N = 999999999999999999999999999999999999999999999999999999999999
MINUS1 = 10000000000000000000000000000000000000000
Prec = 100000000000000000000
In_Pre = 1000000000000000
Lambda = 1000000000000000000000000000000
# N=1000000000000
# N=100000000000000000000
# mod  = np.array(mod1 , dtype=np.uint32)
pp = N

Prec1=1000000000000000000
def mod(x,modulus):
    numer, denom = x.as_numer_denom()
    return numer*mod_inverse(denom,modulus) % modulus
def Input(type):
    if type == 1:
        #file1 = 'X_file_m.csv'
        #file2 = 'y_file_m.csv'
        data1 = pd.read_csv('MPG1.csv')
        data2 = pd.read_csv('MPG3.csv')
        data3 = pd.read_csv('MPG2.csv')
        cols1 = data1.columns.tolist()
        cols1.remove('carname')
        cols1.remove('mpg')
        cols2 = data2.columns.tolist()
        cols2.remove('carname')
        cols2.remove('mpg')
        cols3 = data3.columns.tolist()
        cols3.remove('carname')
        cols3.remove('mpg')
        X1 = data1[cols1].values
        X2 = data2[cols2].values
        X3 = data3[cols3].values
        y1 = data1['mpg'].values
        y2 = data2['mpg'].values
        y3 = data3['mpg'].values
        yM = np.concatenate((y1, y2), axis=0)
        XT1, X_Test1, yT1, y_Test1 = train_test_split(X1, y1, test_size=.20)
        XT2, X_Test2, yT2, y_Test2 = train_test_split(X2, y2, test_size=.20)
        dx = np.concatenate((XT1, XT2), axis=0)  # XTrain
        dy = np.concatenate((yT1, yT2), axis=0)  # y Train
        X_test = np.concatenate((X_Test1, X_Test2), axis=0)
        y_test = np.concatenate((y_Test1, y_Test2), axis=0)
        print("Xtest shape...", X_test.shape)
        print("ytes shape...", y_test.shape)
        print("Xtrain1 shape...", XT1.shape)
        print("Xtrain2 shape...", XT2.shape)
        print("Xtrain shape...",dx.shape)
        print("yshape train",dy.shape)

        np.savetxt('X_Train.txt', dx, delimiter=',')
        np.savetxt('y_Train.txt', dy, delimiter=',')
        np.savetxt('X_Test.txt', X_test, delimiter=',')
        np.savetxt('y_Test.txt', y_test, delimiter=',')
        #dx= pd.read_csv('X_file_m.csv', header=None)

        xd=np.array(dx)
        #print("size of X".xd.shape)
        xd=xd*100
        xd=xd.astype(int)
        np.savetxt('X_file_100.txt', xd, delimiter=',', fmt="%s")
       # dy = pd.read_csv('y_file_m.csv', header=None)
        yd = np.array(dy)
        yd=yd*100
        yd=yd.astype(int)
        np.savetxt('y_file_100.txt', yd, delimiter=',', fmt="%s")
        data = []
        convert = lambda x: str(x).replace('.', '')
        with open('X_file_100.txt') as xfile:
            reader = csv.reader(xfile)
            for row in reader:
                data.append(list(map(convert, row)))

        np.savetxt('datax.txt', data, delimiter=',', fmt="%s")
        data1 = pd.read_csv('datax.txt', header=None)
        #data1=data1
        '''
        Xn = np.array(data1)
        Xnt = Xn.transpose()
        '''
        X = Matrix(data1)
        X = X * Prec1
        print("X", X.shape)
        datay = []
        with open('y_file_100.txt') as yfile:
            reader = csv.reader(yfile)
            for row in reader:
                datay.append(list(map(convert, row)))

        np.savetxt('datay.txt', datay, delimiter=',', fmt="%s")
        data2 = pd.read_csv('datay.txt', header=None)
        '''
        yn = np.array(data2)
        yn=yn
        '''
        y = Matrix(data2)
        y = y * Prec1
        print("y", y.shape)
        XT = X.T
        XTX = XT * X
        print ("XTX",XTX)
        TR = trace(XTX)
        print("Main Trace", TR)
        
        '''
        datacheck=[]
        convert = lambda x: str(x).replace('.', '')
        with open('check.txt') as xfile:
            reader = csv.reader(xfile)
            for row in reader:
                datacheck.append(list(map(convert, row)))

        np.savetxt('check2.txt', datacheck, delimiter=',', fmt="%s")
        data1 = pd.read_csv('check2.txt', header=None)
        data1=np.array(data1)
        data1=data1*Prec
        print(data1)
        '''

        f1, f11 = X.shape
        f2, f22 =y.shape


        X = gf_trunc(X, N)
        X = Matrix(f1, f11, X)
        y = gf_trunc(y, N)
        y = Matrix(f2, f22, y)
       # print("X..", X)
       # print("y..", y)
        #n1=398
        n1 = 259
        n2 = 7

        X1 = zeros(n1, n2)
        X2 = zeros(n1, n2)
        # X1[:,0]=X[:,0]
       # X1[:, 0:3] = X[:, 0:3]
       # print("X1....part",X[0:249,:])
       # print("X2 part...",X[249:328,:])
        X1[0:196,:] = X[0:196,:]
        X2[196:259,:] = X[196:259,:]
        XA = Matrix(X1)

        XAT = XA.T

       # X2[:, 3:7] = X[:, 3:7]
        XB = Matrix(X2)
        XBT = XB.T
       # print ("XA",XA)
       # print ("XAT",XAT)
       # print ("XB",XB)
       # print("XBT",XBT)
        #XTab=XAT+XBT
        #print ("XTab",XTab)
        
        np.savetxt('XA.txt', XA, delimiter=',',fmt="%s")
        np.savetxt('XB.txt', XB, delimiter=',',fmt="%s")
        np.savetxt('XAT.txt', XAT, delimiter=',',fmt="%s")
        np.savetxt('XAT1.txt', XAT, delimiter=',', fmt="%s")
        np.savetxt('XBT.txt', XBT, delimiter=',',fmt="%s")
        np.savetxt('XBT1.txt', XBT, delimiter=',', fmt="%s")
       # random.seed(0)
        YFA = (gf_random(n1 - 1, N, ZZ))
        YFA= Matrix(n1, 1, YFA)

        YFB = (y - YFA)
        YFB = gf_trunc(YFB, N)
        YFB = Matrix(n1, 1, YFB)
      #  print("YFB",YFB)
        np.savetxt('YFA.txt', YFA, delimiter=',',fmt="%s")
        np.savetxt('YFB.txt', YFB, delimiter=',',fmt="%s")
        # print ("XB",XB)
        # print ("All",XA,XAT,YFA,"XB",XB,XBT,YFB)
    elif type == 2:
        Z1A = pd.read_csv('A_Out.txt', header=None)
        Z1A = Matrix(Z1A)
        Z1B = pd.read_csv('B_Out1.txt', header=None)
        Z1B = Matrix(Z1B)
        TRA = trace(Z1A)
        TRA = TRA % N
        TRA = Matrix([[TRA]])
        TRB = trace(Z1B)
        TRB = TRB % N
        TRB = Matrix([[TRB]])

        TR1A = 100000
        TR1A = Matrix([[TR1A]])

        TR1B = 100000
        TR1B = Matrix([[TR1B]])

        np.savetxt('TRA.txt', TRA, delimiter=',',fmt="%s")
        np.savetxt('TRB.txt', TRB, delimiter=',',fmt="%s")
        np.savetxt('TR1A.txt', TR1A, delimiter=',',fmt="%s")
        np.savetxt('TR1B.txt', TR1B, delimiter=',',fmt="%s")
    # return XA, XAT, YFA, XB, XBT, YFB
    elif type == 3:
       # r = 398
        r = 328
        c = 7
        TR1A = pd.read_csv('TR1A.txt', header=None)
        #TR1A = 30000000000000000000000000000000000000000
        TR1A = Matrix(TR1A)
        TR1B = pd.read_csv('TR1B.txt', header=None)
        #TR1B=999999999999999999970000000000000000000000000000026103618567
        TR1B = Matrix(TR1B)
        Z2A = eye(c)
        Z2B = eye(c)
        CaI = TR1A[0] * Z2A
        Z2A = Matrix(CaI)
        CbI = TR1B[0] * Z2B
        Z2B = Matrix(CbI)

        np.savetxt('Z2A.txt', Z2A, delimiter=',',fmt="%s")
        np.savetxt('Z2B.txt', Z2B, delimiter=',',fmt="%s")

#Input(1)
