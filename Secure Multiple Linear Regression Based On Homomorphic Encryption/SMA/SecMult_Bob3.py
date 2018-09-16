import numpy as np
from random import *
from pandas import *
import pandas as pd
#import SecMultInput3
#from SecMultInput3 import Input
import Sec_Mult_Alice_Bob_Input
from Sec_Mult_Alice_Bob_Input import Input
import SecMultTI3
from SecMultTI3 import TI
import SecMult_Sub
from SecMult_Sub import Sub
# import SecMult_Test
# from SecMult_Test import Test
import timeit
import time
from sympy.polys.galoistools import *
from sympy import *
from Client_Server import *
import gmpy2
from gmpy2 import mpz, mpfr, log2

a = 12345678901234567890
gmpy2.get_context().precision = 70
mpz(2 ** log2(a))
mpz(12345678901234567890)
N = 999999999999999999999999999999999999999999999999999999999999
M1 = 10000000000000000000000000000000000000000
Prec = 100000000000000000000
In_Pre = 1000000000000000
Lambda = 1000000000000000000000000000000

def mod(x,modulus):
    numer, denom = x.as_numer_denom()
    return numer*mod_inverse(denom,modulus) % modulus

#XB,XBT,YFB=Input(1)[3:6]
#Input(1)
time.sleep(10)
XB = pd.read_csv('XB.txt', header=None)
XB = Matrix(XB)
XBT = pd.read_csv('XBT.txt', header=None)
XBT = Matrix(XBT)
YFB = pd.read_csv('YFB.txt', header=None)
YFB = Matrix(YFB)

#print("XB,XBT,YFB", XB, XBT, YFB)
r,c = XB.shape
print(XB.shape)
print(XBT.shape)
print(r, c)
C2conn = openEgressSocket('localhost', 10001)  # Open port for sending data
C3conn = openIngressSocket("localhost", 10002)  # Open port for receving data

def Sec_Mult(type, XB, YB, r, c):
    time.sleep(3)
    BB1 = pd.read_csv('BB.csv', header=None)
    BB = Matrix(BB1)
    BB =BB.applyfunc(lambda x: mod(x, N))
    AB = pd.read_csv('AB.csv', header=None)
    AB = Matrix(AB)
    AB = AB.applyfunc(lambda x: mod(x, N))
    C1 = pd.read_csv('C.csv', header=None)
    CB = Matrix(C1)
    CB = CB.applyfunc(lambda x: mod(x, N))
    RB = pd.read_csv('RB.csv', header=None)
    RB =Matrix(RB)
    RPB = pd.read_csv('RPB.csv', header=None)
    RPB = Matrix(RPB)

    XBAB = (XB - AB)

    XBAB = XBAB.applyfunc(lambda x: mod(x, N))

    YBBB = (YB - BB)
    YBBB = YBBB.applyfunc(lambda x: mod(x, N))
   # print("X_Bob", XBAB)
   # print("Y_Bob", YBBB)
    # time.sleep(2)
    sendData(C2conn, XBAB)
    time.sleep(2)
    sendData(C2conn, YBBB)
    clientAlice1 = recvData(C3conn)
    XAAA = Matrix(clientAlice1)  # Reshape in to matrix,
    XAAA = XAAA.applyfunc(lambda x: mod(x, N))
    #print("X_Alice", XAAA)
    clientAlice2 = recvData(C3conn)
    YABA = Matrix(clientAlice2)
    YABA = YABA.applyfunc(lambda x: mod(x, N))
    #print("Y_Alice", YABA)
    clientAlice3 = recvData(C3conn)
    DB = Matrix(clientAlice3)
    DB = DB.applyfunc(lambda x: mod(x, N))
   # print("W", DB)
    clientAlice4 = recvData(C3conn)
    ZA = Matrix(clientAlice4)

    x1 = XAAA * YB
    x2 = XB * YABA
    x3 = XB * YB
    x1 = x1.applyfunc(lambda x: mod(x, N))
    x2 = x2.applyfunc(lambda x: mod(x, N))
    x3 = x3.applyfunc(lambda x: mod(x, N))

    CB = (x1 + x2 + x3 + CB + DB)
    CB = CB.applyfunc(lambda x: mod(x, N))
  #  print("U_BOB", CB)
    np.savetxt('CB_T.txt', CB, delimiter=',', fmt="%s")
    ZA = (ZA + CB)
    ZA = ZA.applyfunc(lambda x: mod(x, N))
    ZA = (ZA + RB)
    ZA = ZA.applyfunc(lambda x: mod(x, N))
    CP = ZA
    ZA1 = ZA.applyfunc(lambda x: mod(x, Prec))
    # print("ZA1",ZA1)
    ZA = ZA1
    ZA = ZA.applyfunc(lambda x: mod(x, N))

    CB = (CB + RPB)
    CB = CB.applyfunc(lambda x: mod(x, N))
    CB = (CB - ZA)
    CB = CB.applyfunc(lambda x: mod(x, N))
    CB = (CB * M1)
    CB = CB.applyfunc(lambda x: mod(x, N))
    #print("B_Out", CB)
    np.savetxt('B_Out.txt', CB, delimiter=',',fmt="%s")
    return (CB)


XTX_B = Sec_Mult(1, XBT, XB, r, c)
XTX_B=Matrix(XTX_B)
XTX_B = XTX_B.applyfunc(lambda x: mod(x, N))
print("XT*X...",XTX_B)
np.savetxt('B_Out1.txt',XTX_B, delimiter=',',fmt="%s")
np.savetxt('Z1B.txt',XTX_B, delimiter=',',fmt="%s")

time.sleep(5)
TRB = pd.read_csv('TRB.txt', header=None)
TRB = Matrix(TRB)
TRB = TRB.applyfunc(lambda x: mod(x, N))
TR1B = pd.read_csv('TR1B.txt', header=None)
TR1B = Matrix(TR1B)
TR1B = TR1B.applyfunc(lambda x: mod(x, N))


for i in range(25):

    TR2_B = (Sec_Mult(2, TRB, TR1B, r, c))
    TR2_B = Matrix (TR2_B)
    TR2B = TR2_B.applyfunc(lambda x: mod(x, N))
    np.savetxt('TR2B.txt', TR2_B, delimiter=',', fmt="%s")
    #Sub(1)
    time.sleep(6)
    TR2B = pd.read_csv('TR2B.txt', header=None)
    TR2B = Matrix(TR2B)
    TR2B = TR2B.applyfunc(lambda x: mod(x, N))

    TR1B = (Sec_Mult(3, TR1B, TR2B, r, c))
    TR1B = Matrix(TR1B)

    TR1B = TR1B.applyfunc(lambda x: mod(x, N))
    print("Cb-TR1B",TR1B)
    np.savetxt('TR1B.txt', TR1B, delimiter=',', fmt="%s")

time.sleep(5)
Z1B = pd.read_csv('Z1B.txt', header=None)

Z1B = Matrix(Z1B)
Z1B = Z1B.applyfunc(lambda x: mod(x, N))
Z2B = pd.read_csv('Z2B.txt', header=None)
Z2B = Matrix(Z2B)
Z2B = Z2B.applyfunc(lambda x: mod(x, N))
#print("Z1B",Z1B)
print ("Z2B...",Z2B)
for i in range(30):
    #print("Z1Binside",i, Z1B)
    #print("Z2Binside...",i, Z2B)
    Z3B = (Sec_Mult(4, Z1B, Z2B, r, c))
    Z3B = Matrix(Z3B)
    Z3B = Z3B.applyfunc(lambda x: mod(x, N))
    #print ("Z3B inside B",Z3B)
    np.savetxt('Z3B.txt', Z3B, delimiter=',', fmt="%s")
    #Sub(1)
    time.sleep(7)
    Z3B = pd.read_csv('Z3B.txt', header=None)
    Z3B = Matrix(Z3B)
    Z3B = Z3B.applyfunc(lambda x: mod(x, N))
    #print("Z3B after sub and before send iteration",i, Z3B)
    Z2B = (Sec_Mult(5, Z2B, Z3B, r, c))
    Z2B = Matrix(Z2B)
    Z2B = Z2B.applyfunc(lambda x: mod(x, N))
    print("Inverse B...", i, Z2B)
    np.savetxt('Z2B.txt', Z2B, delimiter=',', fmt="%s")

print("Inverse B...", i, Z2B)

XBT=pd.read_csv('XBT.txt',header=None)
XBT = Matrix(XBT)
Z2B=pd.read_csv('Z2B.txt',header=None)
Z2B = Matrix(Z2B)
Z2B = Z2B.applyfunc(lambda x: mod(x, N))

#time.sleep(2)
print ("Z2B...",Z2B)
print ("XBT....",XBT)
Z4B= Sec_Mult (6,Z2B,XBT,r,c) # up to (X_Trans*X)_Inverse * X_Trans
np.savetxt('Z4B.txt', Z4B, delimiter=',',fmt="%s")
print ("XT*X_In*XT",Z4B)
#time.sleep(2)
#print ("Z4B...",Z4B)
#print ("YFB....",YFB)
Z5B=Sec_Mult (7,Z4B,YFB,r,c)
np.savetxt('Z5B.txt', Z5B, delimiter=',',fmt="%s")
print ("Beta_B",Z5B)



