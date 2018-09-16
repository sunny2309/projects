import numpy as np
from random import *
from pandas import *
import pandas as pd
import SecMultInput2
from SecMultInput2 import Input
import SecMultTI2
from SecMultTI2 import TI
import SecMult_Sub
from SecMult_Sub import  Sub
from Client_Server import*
import timeit
import time
import numpy as np
from sympy.polys.domains import ZZ
from sympy.polys.galoistools import gf_neg
from sympy.polys.galoistools import *
from sympy import *
import gmpy2
from gmpy2 import mpz,mpfr,log2
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
a=12345678901234567890
gmpy2.get_context().precision=70
mpz(2**log2(a))
mpz(12345678901234567890)
N = 999999999999999999999999999999999999999999999999999999999999
M1 = 10000000000000000000000000000000000000000
Prec = 100000000000000000000
In_Pre = 1000000000000000
Lambda = 1000000000000000000000000000000


def mod(x,modulus):
    numer, denom = x.as_numer_denom()
    return numer*mod_inverse(denom,modulus) % modulus


def Test():

    Z5A=pd.read_csv('Z5A.txt', header=None)
    Z5A = Matrix(Z5A)
    Z5B=pd.read_csv('Z5B.txt', header=None)
    Z5B = Matrix(Z5B)
    TOO=(Z5A+Z5B)
    TOO = TOO.applyfunc(lambda x: mod(x, N))
    np.savetxt('Beta.txt', TOO, delimiter=',', fmt="%s")
    beta=[]
    for i in TOO:
        if ((i) % N < (-i) % N):
            beta.append(i)
           # print (r)
        else:
            r = i-N
            beta.append(r)
           # print ('-',l)


    print (beta)
    beta1=[]
    for i in beta:
        if i>Lambda:
            i=i-M1
           # print (i)
            beta1.append(i)
        else:
            beta1.append(i)
    beta1=(list(map(int,beta1)))

    beta=np.array(beta1)
    beta=beta/Prec
    print(beta)
    np.savetxt('Final_Beta.txt', beta, delimiter=',',fmt='%s')

   # beta = pd.read_csv('Final_Beta.txt', header=None)
    beta=np.array(beta)
    clf = linear_model.LinearRegression(fit_intercept=False)
    X_Train = pd.read_csv('X_Train.txt', header=None)
    y_Train = pd.read_csv('y_Train.txt', header=None)
    X_Test = pd.read_csv('X_Test.txt', header=None)
    y_Test = pd.read_csv('y_Test.txt', header=None)
    X_Train = np.array(X_Train)
    y_Train = np.array(y_Train)
    X_Test = np.array(X_Test)
    y_Test = np.array(y_Test)
    print(y_Test.shape)
    print(X_Test.shape)

    b = clf.fit(X_Train,y_Train)
    b.coef_ = beta.T
    y_pred = b.predict(X_Test)
    print(y_pred.shape)
    mse = mean_squared_error(y_Test, y_pred)
    print ("MSE =", mse)
    mse = np.array([mse])
    np.savetxt('mse.txt', mse, delimiter=',')
#Test()

