import numpy as np
from random import *
from pandas import *
import pandas as pd
import SecMultInput
from SecMultInput import Input
import SecMultTI
from SecMultTI import TI
import SecMult_Alice
from SecMult_Alice import Sec_Mult_A
import SecMult_Bob
from SecMult_Bob import Sec_Mult_B

from Client_Server import*
import AI
from AI import AI
import timeit
import time
import Inverse
from Inverse import modinv

N=100
pp=N
r=2
c=2
XA,XAT,YFA=Input()[0:3]
XB,XBT,YFB=Input()[3:6]

TI(1, r, c)

XTX_A=Sec_Mult_A (1,XAT,XA,r,c)
XTX_B=Sec_Mult_B (1,XBT,XB,r,c)
print("Alice",XTX_A)
print("Bob",XTX_B)