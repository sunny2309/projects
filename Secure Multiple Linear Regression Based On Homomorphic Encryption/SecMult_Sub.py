import numpy as np
from random import *
from pandas import *
import pandas as pd
import SecMultInput3
from SecMultInput3 import Input
import SecMultTI3
from SecMultTI3 import TI
from Client_Server import*
import timeit
import time
from sympy.polys.galoistools import *
from sympy import *
import gmpy2
from gmpy2 import mpz,mpfr,log2
a=12345678901234567890
gmpy2.get_context().precision=70
mpz(2**log2(a))
mpz(12345678901234567890)
N=999999999999999999999999999999999999999999999999999999999999
M1=10000000000000000000000000000000000000000
Prec=100000000000000000000
In_Pre=1000000000000000
Lambda=1000000000000000000000000000000

def mod(x,modulus):
    numer, denom = x.as_numer_denom()
    return numer*mod_inverse(denom,modulus) % modulus
def Sub(type):
    if type==1:
        TR2_A = pd.read_csv('TR2A.txt', header=None)
        TR2_A = Matrix(TR2_A)
        TR2_A = TR2_A.applyfunc(lambda x: mod(x, N))
        Sub_A=TR2_A
        TR2_B = pd.read_csv('TR2B.txt', header=None)
        TR2_B = Matrix(TR2_B)
        TR2_B = TR2_B.applyfunc(lambda x: mod(x, N))
        Sub_B = TR2_B
        A = Sub_A
        B = Sub_B
       # print("A", A)
        #print("B", B)
        A = A.applyfunc(lambda x: mod(x, N))

        B = B.applyfunc(lambda x: mod(x, N))
        Prec1 = Matrix([[Prec]])
        A= -A
        B= -B
        A = (A + Prec1)
        A = A.applyfunc(lambda x: mod(x, N))
        B = (B + Prec1)
        B = B.applyfunc(lambda x: mod(x, N))
        #print ("A",A)
        #print("B",B)
      #  print ("TR2B inside sub protocol",B)
        np.savetxt('TR2A.txt', A, delimiter=',',fmt="%s")
        np.savetxt('TR2B.txt', B, delimiter=',',fmt="%s")
    elif type == 2:

        col=8

        Z3A = pd.read_csv('Z3A.txt', header=None)
        Z3A = Matrix(Z3A )
        Z3A = Z3A.applyfunc(lambda x: mod(x, N))
        Sub_A = Z3A
        Z3B = pd.read_csv('Z3B.txt', header=None)
        Z3B = Matrix(Z3B )
        Z3B = Z3B.applyfunc(lambda x: mod(x, N))
       # print("Z3A inside sub", Z3A)
       # print("Z3B inside sub", Z3B)
        Sub_B = Z3B
        A=Sub_A
        B=Sub_B
        A=(-A)
        A = A.applyfunc(lambda x: mod(x, N))
        B=(-B)
        B = B.applyfunc(lambda x: mod(x, N))
        I=eye(col)
        Prec1=I*Prec

        A=(A+Prec1)
        A = A.applyfunc(lambda x: mod(x, N))
        B=(B+Prec1)
        B = B.applyfunc(lambda x: mod(x, N))
        #print("Z3A inside sub aftre subtraction", A)
        #print("Z3B inside sub after subtraction", B)

        np.savetxt('Z3A.txt', A, delimiter=',',fmt="%s")
        np.savetxt('Z3B.txt', B, delimiter=',',fmt="%s")

