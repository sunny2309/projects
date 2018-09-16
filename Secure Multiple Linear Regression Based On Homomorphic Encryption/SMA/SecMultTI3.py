import numpy as np
from random import *
from pandas import *
import pandas as pd
from Client_Server import*
import timeit
import time
from sympy.polys.domains import ZZ
from sympy.polys.galoistools import gf_neg
from sympy.polys.galoistools import *
from sympy import *
import gmpy2
from gmpy2 import mpz,mpfr,log2



N=999999999999999999999999999999999999999999999999999999999999
MINUS1=10000000000000000000000000000000000000000
Prec=100000000000000000000
In_Pre=1000000000000000
Lambda=1000000000000000000000000000000

def mod(x,modulus):
    numer, denom = x.as_numer_denom()
    return numer*mod_inverse(denom,modulus) % modulus

def TI(type, row, col):
    if type == 1:
        dim1=col
        dim2=row
        dim3=col
    elif type== 2:
        dim1 = dim2 = dim3 = 1
    elif type == 3:
        dim1 = dim2 = dim3 = 1
    elif type== 4:
        dim1=dim2=dim3=col
    elif type== 5:
        dim1=dim2=dim3=col
    elif type== 6:
        dim1=dim2=col
        dim3=row
    elif type== 7:
        dim1=col
        dim2=row
        dim3=1

    d1=dim1* dim2
    d2=dim2* dim3
    d3=dim1 *dim3
   # print("d1,d2,d3",d1,d2,d3)
    AA =(gf_random(d1-1, N, ZZ))
    AA = Matrix(dim1,dim2, AA)

    BA = (gf_random(d2-1, N, ZZ))
    BA = Matrix(dim2, dim3, BA)

    T = (gf_random(d3-1, N, ZZ))
    T= Matrix(dim1, dim3, T)

    AB = (gf_random(d1-1, N, ZZ))
    AB = Matrix(dim1, dim2, AB)

    BB = (gf_random(d2-1, N, ZZ))
    BB = Matrix(dim2, dim3, BB)

    x1 = AA * BB
    x1 = gf_trunc(x1, N)
    x1 = Matrix(dim1, dim3, x1)

    x2 = AB * BA
    x2 = gf_trunc(x2, N)
    x2 = Matrix(dim1, dim3, x2)

    C=x1+x2-T
    C = gf_trunc(C, N)
    C=Matrix(dim1,dim3,C)

    #print ("C",C)
   # print ("AA",AA)
    #print ("shapebefore",AA.shape)

    RP = (gf_random(d3 - 1, Prec, ZZ))
    RP = gf_trunc(RP, Prec)
    RP = Matrix(dim1, dim3, RP)
    RPP = (gf_random(d3 - 1,Lambda , ZZ))
    RPP = gf_trunc(RPP, Lambda)
    RPP = Matrix(dim1, dim3, RPP)
    R= RPP * Prec + RP
    R = gf_trunc(R, N)
    R = Matrix(dim1, dim3, R)

    RA = (gf_random(d3 - 1, N, ZZ))
    RA = gf_trunc(RA, N)
    RA = Matrix(dim1, dim3, RA)
    RPA = (gf_random(d3 - 1, N, ZZ))
    RPA = gf_trunc(RPA, N)
    RPA = Matrix(dim1, dim3, RPA)
    RB=R-RA
    RPB=RP-RPA
    RB = RB.applyfunc(lambda x: mod(x, N))
    RPB = RPB.applyfunc(lambda x: mod(x, N))


    np.savetxt('AA.csv', AA, delimiter=',',fmt="%s")
    np.savetxt('BA.csv', BA, delimiter=',',fmt="%s")
    np.savetxt('T.csv', T, delimiter=',',fmt="%s")
    np.savetxt('AB.csv', AB, delimiter=',',fmt="%s")
    np.savetxt('BB.csv', BB, delimiter=',',fmt="%s")
    np.savetxt('C.csv', C, delimiter=',',fmt="%s")



    np.savetxt('RA.csv', RA, delimiter=',',fmt="%s")
    np.savetxt('RPA.csv', RPA, delimiter=',',fmt="%s")
    np.savetxt('RB.csv', RB, delimiter=',',fmt="%s")
    np.savetxt('RPB.csv', RPB, delimiter=',',fmt="%s")

    #return XA, XAT, YFA, XB, XBT, YFB
#TI(1,10,20)