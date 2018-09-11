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
import SecMult_Test
from SecMult_Test import Test
from Client_Server import *
from sympy.polys.galoistools import *
from sympy import *
import timeit
import time
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

pp = N
# np.random.seed()

def mod(x,modulus):
    numer, denom = x.as_numer_denom()
    return numer*mod_inverse(denom,modulus) % modulus

#XA,XAT,YFA=Input(1)[0:3]
Input(1)
XA = pd.read_csv('XA.txt', header=None)
XA = Matrix(XA)
XAT = pd.read_csv('XAT.txt', header=None)
XAT = Matrix(XAT)
YFA = pd.read_csv('YFA.txt', header=None)
YFA = Matrix(YFA)
# XA,XAT,YFA=Input()[0:3]  # ALice
# print ("XA,XAT,YFA",XA,XAT,YFA)

r,c = XA.shape

C1conn = openIngressSocket("localhost", 10001)  # Open port for receving data
time.sleep(1)
C2conn = openEgressSocket('localhost', 10002)  # Open port for sending data

def Sec_Mult(type, XA, YA, r, c):
    print("r11,c,type", r, c, type)
    TI(type, r, c)
    # AA, BA, T = TI(type, r, c)[0:3]  # ALice

    AA1 = pd.read_csv('AA.csv', header=None)
    AA = Matrix(AA1)
    AA = AA.applyfunc(lambda x: mod(x, N))
    BA1 = pd.read_csv('BA.csv', header=None)
    BA = Matrix(BA1)
    BA= BA.applyfunc(lambda x: mod(x, N))
    CA = pd.read_csv('T.csv', header=None)
    CA = Matrix(CA)
    CA = CA.applyfunc(lambda x: mod(x, N))
    RA = pd.read_csv('RA.csv', header=None)
    RA = Matrix(RA)
    RA = RA.applyfunc(lambda x: mod(x, N))
    RPA = pd.read_csv('RPA.csv', header=None)
    RPA = Matrix(RPA)
    RPA = RPA.applyfunc(lambda x: mod(x, N))

    clientBob1 = recvData(C1conn)
    XBAB = Matrix(clientBob1)
    XBAB = XBAB.applyfunc(lambda x: mod(x, N))
    XA = XA.applyfunc(lambda x: mod(x, N))
    YA = YA.applyfunc(lambda x: mod(x, N))
   # print("XA", XA)
   # print("YA", YA)
    # XAYA=(np.dot(XA,YA))%N
    # print("XAYA",XAYA)
    #print("XB_AB", XBAB)
    clientBob2 = recvData(C1conn)
    YBBB = Matrix(clientBob2)
    YBBB = YBBB.applyfunc(lambda x: mod(x, N))
    #print("YB_BB", YBBB)
    # print("clientBob1",clientBob1)
    # print ("X_Bob,Y_Bob",X_Bob,Y_Bob)
    # Con_Bob = recvData(C1conn,X_Bob,Y_Bob)  #Receive function from Bob
    m, m2 = XBAB.shape
    m3, n = BA.shape

    d3 = m * n
    DA = (gf_random(d3 - 1, N, ZZ))
    DA = Matrix(m, n, DA)

    XAAA = (XA - AA)
    XAAA = XAAA.applyfunc(lambda x: mod(x, N))
    #print("XAAA", XAAA)

    YABA = (YA - BA)
    YABA = YABA.applyfunc(lambda x: mod(x, N))
    #print("YABA", YABA)
    x1=AA* YBBB
    x2=XBAB* BA
    x3=XA* YA
    x1 = x1.applyfunc(lambda x: mod(x, N))
    x2 = x2.applyfunc(lambda x: mod(x, N))
    x3 = x3.applyfunc(lambda x: mod(x, N))


    DB = (x1 + x2 + x3 - DA)
    DB = DB.applyfunc(lambda x: mod(x, N))

    sendData(C2conn, XAAA)  # Sending to Bob Xalice,Yalice,W
    time.sleep(2)
    sendData(C2conn, YABA)
    time.sleep(2)
    sendData(C2conn, DB)

    CA = (CA + DA)
    CA= CA.applyfunc(lambda x: mod(x, N))
    #print("T_Tot", CA)
    np.savetxt('CA_T.txt', CA, delimiter=',', fmt="%s")

    # CA1 = (CA+DA) % N
    # print ("Actual A_out",CA1)
    RA = (RA + CA)
    RA = RA.applyfunc(lambda x: mod(x, N))
    time.sleep(2)
    sendData(C2conn, RA)
    # A_Out1=(T+T1+RA)%N

    CA = (CA + RPA)
    CA = CA.applyfunc(lambda x: mod(x, N))
    CA = (CA * M1)
    CA = CA.applyfunc(lambda x: mod(x, N))
    A_Out = CA
   # print("A_Out", A_Out)
    np.savetxt('A_Out.txt', A_Out, delimiter=',',fmt="%s")

    # A_Out=CA
    return (A_Out)


'''
XAT=pd.read_csv('XAT.txt',header=None)
XAT = Matrix(XAT)
Z2A=pd.read_csv('Z2A2.txt',header=None)
Z2A = Matrix(Z2A)
Z2A = Z2A.applyfunc(lambda x: mod(x, N))
print("Z2A..",Z2A)
print ("XAT",XAT)
#time.sleep(1)
Z4A= Sec_Mult (6,Z2A,XAT,r,c) # up to (X_Trans*X)_Inverse * X_Trans
#np.savetxt('Z4A.txt', Z4A, delimiter=',',fmt="%s")
print ("XT*X_In*XT",Z4A)
#print ("Z4A..",Z4A)
#print("YFA",YFA)
time.sleep(1)
Z5A=Sec_Mult (7,Z4A,YFA,r,c)
#np.savetxt('Z5A.txt', Z5A, delimiter=',',fmt="%s")
print ("Beta_A",Z5A)

'''
start = timeit.default_timer()
start1 = timeit.default_timer()
XTX_A = Sec_Mult(1, XAT, XA, r, c)
XTX_A =Matrix(XTX_A)
XTX_A = XTX_A.applyfunc(lambda x: mod(x, N))
np.savetxt('Z1A.txt',XTX_A, delimiter=',',fmt="%s")
print("XT*X...",XTX_A)
stop1 = timeit.default_timer()
print("Multiplication time..",stop1-start1)
time.sleep(2)
Input(2)

TRA = pd.read_csv('TRA.txt', header=None)
TRA = Matrix(TRA)
TRA = TRA.applyfunc(lambda x: mod(x, N))
TR1A = pd.read_csv('TR1A.txt', header=None)
TR1A = Matrix(TR1A)
TR1A = TR1A.applyfunc(lambda x: mod(x, N))
for i in range(25):

    #print("TRA..inside loop",TRA)
    #print("TRA..inside loop",TR1A)
    TR2_A = (Sec_Mult(2, TRA, TR1A, r, c))
    TR2_A = Matrix(TR2_A)
    TR2A = TR2_A.applyfunc(lambda x: mod(x, N))
    #print ("TR1A..",TR1A)
    #print ("TR2_A..",i,TR2A)
    np.savetxt('TR2A.txt', TR2_A, delimiter=',',fmt="%s")
    time.sleep(3)
    Sub(1)
    TR2A = pd.read_csv('TR2A.txt', header=None)
    TR2A = Matrix(TR2A)
    TR2A = TR2A.applyfunc(lambda x: mod(x, N))
   # print("TR1Abefore snd itern", TR1A)
   # print("TRA2 after sub and before send iteration", TR2A)
    TR1A = (Sec_Mult(3, TR1A, TR2A, r, c))
    TR1A = Matrix(TR1A)
    TR1A = TR1A.applyfunc(lambda x: mod(x, N))
    print("Ca...",i,TR1A)
    np.savetxt('TR1A.txt', TR1A, delimiter=',',fmt="%s")

time.sleep(2)
Input(3)
Z1A = pd.read_csv('Z1A.txt', header=None)
Z1A = Matrix(Z1A)
Z1A = Z1A.applyfunc(lambda x: mod(x, N))
Z2A = pd.read_csv('Z2A.txt', header=None)
Z2A = Matrix(Z2A)
Z2A = Z2A.applyfunc(lambda x: mod(x, N))
print ("Z1A",Z1A)
print("Z2A..",Z2A)
for i in range(30):
    #print("Z1Ainside..",i, Z1A)
    #print("Z2Ainside..",i, Z2A)
    Z3A = (Sec_Mult(4, Z1A, Z2A, r, c))
    Z3A = Matrix(Z3A )
    Z3A = Z3A .applyfunc(lambda x: mod(x, N))
    np.savetxt('Z3A.txt', Z3A, delimiter=',',fmt="%s")
    time.sleep(4)
    Sub(2)
    Z3A = pd.read_csv('Z3A.txt', header=None)
    Z3A = Matrix(Z3A)
    Z3A = Z3A.applyfunc(lambda x: mod(x, N))
    #print("Z3Abefore 2nd itern", Z2A)
    #print("Z3A after sub and before send iteration  ",i, Z3A)
    Z2A = (Sec_Mult(5, Z2A, Z3A, r, c))
    Z2A = Matrix(Z2A)
    Z2A = Z2A.applyfunc(lambda x: mod(x, N))
    print("Inverse A...", i, Z2A)
    np.savetxt('Z2A.txt', Z2A, delimiter=',',fmt="%s")
print("Inverse A...",i,Z2A)

XAT=pd.read_csv('XAT.txt',header=None)
XAT = Matrix(XAT)
Z2A=pd.read_csv('Z2A.txt',header=None)
Z2A = Matrix(Z2A)
Z2A = Z2A.applyfunc(lambda x: mod(x, N))
print("Z2A..",Z2A)
print ("XAT",XAT)
#time.sleep(1)
Z4A= Sec_Mult (6,Z2A,XAT,r,c) # up to (X_Trans*X)_Inverse * X_Trans
np.savetxt('Z4A.txt', Z4A, delimiter=',',fmt="%s")
print ("XT*X_In*XT",Z4A)
print ("Z4A..",Z4A)
print("YFA",YFA)
time.sleep(1)
Z5A=Sec_Mult (7,Z4A,YFA,r,c)
np.savetxt('Z5A.txt', Z5A, delimiter=',',fmt="%s")
print ("Beta_A",Z5A)
stop = timeit.default_timer()

time.sleep(5)
Test()
#print("Time taken ...", stop-start)


# stop = timeit.default_timer()

# print("Time taken by Alice",stop-start)

