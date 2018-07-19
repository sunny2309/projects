## csp.py
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

def compute_twos_complement(binary):
    out = ''
    for i in binary:
        out += '1' if i=='0' else '0'
    #print(out)
    return Add()(out,((len(binary)-1)*'0')+'1')
        
def float_to_bin(num,size=24):
    twos_complement = True if num<0 else False
    int_p,frac_p = str(num).split('.')
    frac_p = float('0.'+frac_p)
    int_b = bin(int(int_p))[2:]
    frac_b = '.'
    for i in range(8):
        #print(frac_p)
        frac_p = frac_p*2
        t_i, t_f = str(frac_p).split('.')
        frac_b += t_i
        if t_i == '1':
            frac_p -= 1.
        #print(frac_p)
    out = int_b+frac_b
    out = out.replace('.','')
    #print(out)
    out = out[:-4]  ## Need to change to -8 in future
    out += '11111100'
    out = out.zfill(size)
    if twos_complement:
        temp_out = compute_twos_complement(out[:size-8])
    out = temp_out+out[size-8:] if twos_complement else out
    #print(out)
    return out

def int_to_bin(num):
    twos_complement = True if num<0 else False
    int_b = bin(int(num))[2:]
    if twos_complement:
        int_b = compute_twos_complement(int_b)
    return int_b

def bin_to_float(binary):
    int_p , frac_p = binary[:len(binary)-8], binary[len(binary)-8:]
    int_part = 0.0
    for i,j in enumerate(reversed(int_p[1:])):
        int_part += (int(j) * pow(2, i))
    #print(int_part)
    int_part -= int(binary[0])*pow(2,len(binary)-9) ## 15
    #print(int_part)
    pow_part = 0.0
    for i,j in enumerate(reversed(frac_p[1:])):
        pow_part += (int(j) * pow(2, i))
    pow_part -= pow(2,7)
    #print(pow_part)
    return int_part*pow(2,pow_part)

def bin_to_int(binary):
    int_part = 0
    for i,j in enumerate(reversed(binary[1:])):
        int_part += (int(j) * pow(2, i))
    int_part -= int(binary[0])*pow(2,len(binary)-1)
    return int_part
    
def bin_to_int_garbled(binary,wire_labels):
    wl = dict(zip(wire_labels.values(),wire_labels.keys()))
    b2 = ''.join([wl[i].split('_')[-1] for i in binary])
    int_part = 0
    for i,j in enumerate(reversed(b2[1:])):
        int_part += (int(j) * pow(2, i))
    int_part -= int(wl[binary[0]].split('_')[-1])*pow(2,len(binary)-1)
    return int_part

def compute_twos_complement_garbled(binary,wire_labels={}):
    wl = dict(zip(wire_labels.values(),wire_labels.keys()))
    out = []
    for i in binary:
        out += [wire_labels['B_1']] if '0' in wl[i] else [wire_labels['B_0']]
    #print(out)
    out = GarbledAdd()(((len(binary)-1)*[wire_labels['A_0']])+[wire_labels['A_1']], out, wire_labels)
    wl = dict(zip(wire_labels.values(),wire_labels.keys()))
    out = [wire_labels[wl[i].replace('C','B')] for i in out]
    return out

class GarbledLeftShift(Gate):
    def __call__(self,a, n, wire_labels={}, garbled_table=[]):
        #print(a,wire_labels)
        wl = dict(zip(wire_labels.values(), wire_labels.keys()))
        val =  [j for j in [wl[i] for i in list(set(a))] if '0' in j][0]
        return a[n:] + [wire_labels[val]] * n

class GarbledRightShift(Gate):
    def __call__(self,a, n, wire_labels={}, garbled_table=[]):
        wl = dict(zip(wire_labels.values(), wire_labels.keys()))
        val =  [j for j in [wl[i] for i in list(set(a))] if '0' in j][0]
        return [wire_labels[val]] * n + a[:-n]

class GarbledGreaterThanEqualTo(Gate):
    def __call__(self,a, b, wire_labels):
        return bin_to_int_garbled(a, wire_labels) >= bin_to_int_garbled(b, wire_labels)
    
class GarbledPadZerosLeft(Gate):
    def __call__(self,a, n, wire_labels={}, garbled_table=[]):
        wl = dict(zip(wire_labels.values(), wire_labels.keys()))
        val =  [j for j in [wl[i] for i in list(set(a))] if '0' in j][0]
        return [wire_labels[val]] * n + a 

class GarbledPadZerosRight(Gate):
    def __call__(self,a, n, wire_labels={}, garbled_table=[]):
        wl = dict(zip(wire_labels.values(), wire_labels.keys()))
        val =  [j for j in [wl[i] for i in list(set(a))] if '0' in j][0]
        return a + [wire_labels[val]] * n
    
class GarbledAnd(Gate):
    def __call__(self,a,b,wire_labels={},garbled_table=[]):
        
        des_b_0 = DES.new(wire_labels['B_0'], DES.MODE_ECB)
        des_a_0 = DES.new(wire_labels['A_0'], DES.MODE_ECB)
        des_b_1 = DES.new(wire_labels['B_1'], DES.MODE_ECB)
        des_a_1 = DES.new(wire_labels['A_1'], DES.MODE_ECB)
        ## 0 AND 0 = 0
        
        garbled_table.append(des_a_0.encrypt(des_b_0.encrypt(wire_labels['C_0'])))
        garbled_table.append(des_a_0.encrypt(des_b_1.encrypt(wire_labels['C_0'])))
        garbled_table.append(des_a_1.encrypt(des_b_0.encrypt(wire_labels['C_0'])))
        garbled_table.append(des_a_1.encrypt(des_b_1.encrypt(wire_labels['C_1'])))
        
        des_in = DES.new(b, DES.MODE_ECB)
        des_out = DES.new(a, DES.MODE_ECB)
        for g in garbled_table:
            try:
                deciphered = des_in.decrypt(des_out.decrypt(g)).decode()
            except UnicodeDecodeError:
                continue
            if deciphered in [ wire_labels['C_0'], wire_labels['C_1'] ]:
                return g
        
class GarbledOr(Gate):
    def __call__(self,a,b,wire_labels={},garbled_table=[]):
        
        des_b_0 = DES.new(wire_labels['B_0'], DES.MODE_ECB)
        des_a_0 = DES.new(wire_labels['A_0'], DES.MODE_ECB)
        des_b_1 = DES.new(wire_labels['B_1'], DES.MODE_ECB)
        des_a_1 = DES.new(wire_labels['A_1'], DES.MODE_ECB)
        ## 0 AND 0 = 0
        
        garbled_table.append(des_a_0.encrypt(des_b_0.encrypt(wire_labels['C_0'])))
        garbled_table.append(des_a_0.encrypt(des_b_1.encrypt(wire_labels['C_1'])))
        garbled_table.append(des_a_1.encrypt(des_b_0.encrypt(wire_labels['C_1'])))
        garbled_table.append(des_a_1.encrypt(des_b_1.encrypt(wire_labels['C_1'])))
        
        des_in = DES.new(b, DES.MODE_ECB)
        des_out = DES.new(a, DES.MODE_ECB)
        for g in garbled_table:
            try:
                deciphered = des_in.decrypt(des_out.decrypt(g)).decode()
            except UnicodeDecodeError:
                continue
            if deciphered in [ wire_labels['C_0'], wire_labels['C_1'] ]:
                return g

class GarbledXor(Gate):
    def __call__(self,a,b,wire_labels={},garbled_table=[]):
        
        des_b_0 = DES.new(wire_labels['B_0'], DES.MODE_ECB)
        des_a_0 = DES.new(wire_labels['A_0'], DES.MODE_ECB)
        des_b_1 = DES.new(wire_labels['B_1'], DES.MODE_ECB)
        des_a_1 = DES.new(wire_labels['A_1'], DES.MODE_ECB)
        ## 0 AND 0 = 0
        
        garbled_table.append(des_a_0.encrypt(des_b_0.encrypt(wire_labels['C_0'])))
        garbled_table.append(des_a_0.encrypt(des_b_1.encrypt(wire_labels['C_1'])))
        garbled_table.append(des_a_1.encrypt(des_b_0.encrypt(wire_labels['C_1'])))
        garbled_table.append(des_a_1.encrypt(des_b_1.encrypt(wire_labels['C_0'])))
        #print(garbled_table)
        des_1 = DES.new(a, DES.MODE_ECB)
        des_2 = DES.new(b, DES.MODE_ECB)
        for g in garbled_table:
            try:
                deciphered = des_2.decrypt(des_1.decrypt(g)).decode()
            except UnicodeDecodeError:
                continue
            #print(deciphered)
            #print([ wire_labels['C_0'], wire_labels['C_1'] ])
            if deciphered in [ wire_labels['C_0'], wire_labels['C_1'] ]:
                return g
        
class GarbledNot(Gate):
    def __call__(self,b,wire_labels={},garbled_table=[]):
        
        des_a_0 = DES.new(wire_labels['A_0'], DES.MODE_ECB)
        des_a_1 = DES.new(wire_labels['A_1'], DES.MODE_ECB)
        
        garbled_table.append(des_a_0.encrypt(wire_labels['C_1']))
        garbled_table.append(des_a_1.encrypt(wire_labels['C_0']))
        
        des = DES.new(b, DES.MODE_ECB)
        for g in garbled_table:
            try:
                deciphered = des.decrypt(g).decode()
            except UnicodeDecodeError:
                continue
            if deciphered in [ wire_labels['C_0'], wire_labels['C_1'] ]:
                return g
            
class GarbledAdd(Gate):
    
    def add(self,a,b,carry,wire_labels):
        wl = dict(zip(wire_labels.values(), wire_labels.keys()))
        
        des_in = DES.new(b, DES.MODE_ECB)
        des_out = DES.new(a, DES.MODE_ECB)
        
        o1 = GarbledXor()(a,b,wire_labels)
        #print(a,b,o1)
        o1 = des_in.decrypt(des_out.decrypt(o1)).decode()
        
        o2 = GarbledAnd()(a,b,wire_labels)
        o2 = des_in.decrypt(des_out.decrypt(o2)).decode()
        
        o1 = wire_labels[wl[o1].replace('C','B')]
        
        des_in = DES.new(o1, DES.MODE_ECB)
        des_out = DES.new(carry, DES.MODE_ECB)
        
        add = GarbledXor()(carry,o1,wire_labels)
        add = des_in.decrypt(des_out.decrypt(add)).decode()
        
        o4 = GarbledAnd()(carry,o1,wire_labels)
        o4 = des_in.decrypt(des_out.decrypt(o4)).decode()
        
        o4 = wire_labels[wl[o4].replace('C', 'A')]
        o2 = wire_labels[wl[o2].replace('C', 'B')]
        
        des_in = DES.new(o2, DES.MODE_ECB)
        des_out = DES.new(o4, DES.MODE_ECB)
        
        cary = GarbledOr()(o4, o2, wire_labels)
        cary = des_in.decrypt(des_out.decrypt(cary)).decode()
        
        cary = wire_labels[wl[cary].replace('C', 'A')]
        return str(add), cary
    
    def __call__(self,a, b, wire_labels={}):
        wl = dict(zip(wire_labels.values(),wire_labels.keys()))
        carry = wire_labels['A_0']
        out = []
        for idx,i in reversed(list(enumerate(zip(a[:16],b[:16])))):
            o,carry = self.add(i[0], i[1], carry, wire_labels)
            out.append(o)
        #print(out)
        out = list(reversed(out))
        return out + [ wire_labels[wl[i].replace('A','C')] for i in a[16:]]

class GarbledSubtract(Gate):
    
    def __call__(self,a,b, wire_labels={}):
        b2 = compute_twos_complement_garbled(b[:16],wire_labels)
        #print(b2,b[16:])
        b2 = list(b2) + list(b[16:])
        return GarbledAdd()(a,b2,wire_labels)

class GarbledMultiplication(Gate):

    def __call__(self,a,b, wire_labels={}):
        wl = dict(zip(wire_labels.values(),wire_labels.keys()))
        s = [wire_labels['B_0']]*16
        m = copy.copy(a[:16])
        for i in reversed(b[:16]):
            #print(m,s)
            if '1' in wl[i]:     # when digit is one, add the intermediary product
                s = GarbledAdd()(m, s, wire_labels)
                s = [ wire_labels[wl[i].replace('C','B')] for i in s]
            m = GarbledLeftShift()(m, 1, wire_labels)  # shift one per digit in b
        #exponent = Add()(a[16:],b[16:])
        s = [ wire_labels[wl[i].replace('B','C')] for i in s]
        s = GarbledRightShift()(s,4,wire_labels) ## Shifting left by 4 places to get exponent back to 2^-4
        return s+[ wire_labels[wl[val].replace('B', 'C')] for val in b[16:]]
        
class GarbledDivision(Gate):
    
    def __call__(self,a, b,wire_labels={}):
        wl = dict(zip(wire_labels.values(),wire_labels.keys()))
        reminder,reminder_prev = GarbledSubtract()(a[:16], b[:16], wire_labels), copy.copy(a[:16])
        quotient = GarbledAdd()([wire_labels['A_0']]*16, ([wire_labels['B_0']] * 15)+[wire_labels['B_1']], wire_labels)
        reminder = [ wire_labels[wl[val].replace('C', 'A')] for val in reminder]
        while wire_labels['A_1'] in reminder:
            reminder_prev = copy.copy(reminder)
            reminder = GarbledSubtract()(reminder, b[:16], wire_labels)
            reminder = [ wire_labels[wl[val].replace('C', 'A')] for val in reminder]
            #print(bin_to_int(reminder))
            #quotient = Add()(quotient, '00000001')
            if (bin_to_int_garbled(reminder,wire_labels) < 0 and bin_to_int_garbled(reminder_prev,wire_labels) > 0) or \
                (bin_to_int_garbled(reminder,wire_labels) > 0 and bin_to_int_garbled(reminder_prev,wire_labels) < 0):
                break
            quotient = [ wire_labels[wl[val].replace('C', 'A')] for val in quotient]
            quotient = GarbledAdd()(quotient, ([wire_labels['B_0']] * 15)+[wire_labels['B_1']], wire_labels)
        quotient = GarbledLeftShift()(quotient,4, wire_labels)
        return quotient+[ wire_labels[wl[val].replace('A', 'C')] for val in a[16:]]
    
class GarbledSqrt(Gate):
    
    def __call__(self,x,wire_labels={}):
        wl = dict(zip(wire_labels.values(), wire_labels.keys()))
        num = copy.copy(x[:16])
        xx = bin_to_int_garbled(x[:16],wire_labels)
        #e = [i for i in range(int(xx)) if pow(4,i)<=xx][-1]
        e = [i for i in range(int(xx)) if pow(4,i)<=xx]
        #e = e[-1] if e else 0
        e = int_to_bin(pow(4,e[-1])) if e else '0'*len(num)
        e = [wire_labels['B_'+str(i)] for i in e]
        if len(e) < len(num):
            e = GarbledPadZerosLeft()(e,len(num) - len(e),wire_labels)
        r = [ wire_labels['A_0'] ]*len(e)
        while wire_labels['B_1'] in e:
            #temp = Add()(r, e, wire_labels)
            #temp = [ wire_labels[wl[i].replace('C','A')] for i in temp]
            #print(r,e)
            if GarbledGreaterThanEqualTo()(num, GarbledAdd()(r, e, wire_labels), wire_labels):
                #print('IF')
                out = GarbledAdd()(r, e, wire_labels)
                out = [ wire_labels[wl[i].replace('C','B')] for i in out]
                num = GarbledSubtract()(num, out,wire_labels)
                num = [ wire_labels[wl[i].replace('C','A')] for i in num]
                r = GarbledAdd()(GarbledRightShift()(r, 1, wire_labels), e, wire_labels)
                #print(r)
                r = [ wire_labels[wl[i].replace('C','A')] for i in r]
                #print(r)
            else:
                #print('ELSE')
                r = GarbledRightShift()(r, 1, wire_labels)
            e = GarbledRightShift()(e, 2, wire_labels)
        r = GarbledLeftShift()(r,2, wire_labels)
        return r + [ wire_labels[wl[i].replace('A','C')] for i in x[16:] ]
    
class GarbledCircuit(object):
    def __init__(self):
        pass
        
    def convert_to_float(self,a,two_dimension=True):
        out = np.empty(a.shape[:-1],dtype=np.float)
        wl = dict(zip(self.wire_labels.values(),self.wire_labels.keys()))
        if two_dimension:
            for i in range(a.shape[0]):
                for j in range(a.shape[1]):
                    out[i][j] = bin_to_float(''.join([ wl[k].split('_')[-1] for k in a[i][j] ]))
        else:
            for i in range(a.shape[0]):
                out[i] = bin_to_float(''.join([ wl[k].split('_')[-1] for k in a[i] ]))
                
        return out
    
    def calculate_A_b(self):
        ## Subtract mu_A and mu_B from A_hat and b_hat
        self.A_garbled = np.empty(self.A_hat_garbled.shape,dtype=np.object)
        self.b_garbled = np.empty(self.b_hat_garbled.shape,dtype=np.object)
        #print(self.A_hat_garbled.shape,self.b_hat_garbled.shape,self.mu_A_garbled.shape,self.mu_b_garbled.shape)
        for i in range(self.A_hat_garbled.shape[0]):
            for j in range(self.A_hat_garbled.shape[1]):
                print(i,j)
                #print(self.A_hat_garbled[i][j],self.mu_A_garbled[i][j])
                self.A_garbled[i][j] = np.array(GarbledSubtract()(self.A_hat_garbled[i][j],self.mu_A_garbled[i][j],self.wire_labels))
        
        for i in range(self.b_hat_garbled.shape[0]):
            print(i)
            self.b_garbled[i] = np.array(GarbledSubtract()(self.b_hat_garbled[i],self.mu_b_garbled[i], self.wire_labels))
        
        A = self.convert_to_float(self.A_garbled)
        b = self.convert_to_float(self.b_garbled,False)
        print('A',A.shape,A)
        print('B',b.shape,b)
        
    def calculate_L(self):
        ## Get L using A with algo cholesky
        wl = dict(zip(self.wire_labels.values(), self.wire_labels.keys()))
        self.d = self.A_garbled.shape[0]
        self.L = self.A_garbled.copy()
        for i in range(self.d):
            for j in range(self.d):
                self.L[i][j] = np.array([self.wire_labels[wl[i].replace('C','A')] for i in self.L[i][j]])
        #L = np.zeros(A.shape)
        for i in range(self.d-1):
            for j in range(i+1,self.d):
                self.L[i][j] = np.array([ self.wire_labels['A_0'] ] *24)
        for j in range(self.d):
            print(j)
            for k in range(j-1):
                for i in range(j,self.d):
                    ljk = np.array([self.wire_labels[wl[i].replace('A','B')] for i in self.L[j][k]])
                    mul = GarbledMultiplication()(self.L[i][k], ljk,self.wire_labels)
                    mul = np.array([self.wire_labels[wl[i].replace('C','B')] for i in mul])
                    self.L[i][j] = GarbledSubtract()(self.L[i][j], mul, self.wire_labels) ## Need to bring -4 as exponent
                    self.L[i][j] = np.array([self.wire_labels[wl[i].replace('C','A')] for i in self.L[i][j]])
        
            self.L[j][j] = GarbledSqrt()(self.L[j][j],self.wire_labels) ## Need to bring -4 as exponent
            self.L[j][j] = np.array([self.wire_labels[wl[i].replace('C','A')] for i in self.L[j][j]])
            
            for k in range(j+1,self.d):
                ljj = np.array([self.wire_labels[wl[i].replace('A','B')] for i in self.L[j][j]])
                self.L[k][j] = GarbledDivision()(self.L[k][j], ljj, self.wire_labels) ## Need to bring -4 as exponent
                self.L[k][j] = np.array([self.wire_labels[wl[i].replace('C','A')] for i in self.L[k][j]])
                
        LF = self.convert_to_float(self.L)
        print(LF.shape,LF)
        
    def calculate_Y(self):
        
        ## Use back substitution algo to get individual values of Y    : L.T * Y = b
        LT = self.L.transpose(1,0,2)
        self.Y  = np.empty(self.b_garbled.shape,dtype=np.object)
        self.Y[d-1] = GarbledDivision()(self.b_garbled[d-1], LT[d-1][d-1], self.wire_labels)
        self.Y[d-1] = np.array([self.wire_labels[wl[i].replace('C','B')] for i in self.Y[d-1]])
        for i in reversed(range(self.d-1)):
            print(i)
            s = [ self.wire_labels['A_0'] ] * 24
            for j in range(i+1,self.d):
                temp_lij = np.array([self.wire_labels[wl[i].replace('C','A')] for i in LT[i][j]])
                mul = GarbledMultiplication()(temp_lij, self.Y[j], self.wire_labels)
                mul = np.array([self.wire_labels[wl[i].replace('C','B')] for i in mul])
                
                s = GarbledAdd()(s, mul, self.wire_labels)
                s = np.array([self.wire_labels[wl[i].replace('C','A')] for i in s])
                
            s = np.array([self.wire_labels[wl[i].replace('A','B')] for i in s])
            bi = np.array([self.wire_labels[wl[i].replace('C','A')] for i in self.b_garbled[i]])
            self.Y[i] = GarbledDivision()(bi, s, self.wire_labels)
            self.Y[i] = np.array([self.wire_labels[wl[i].replace('C','B')] for i in self.Y[i]])
        
        YF = self.convert_to_float(self.Y,False)
        print(YF.shape,YF)
        
    
    def calculate_beta(self):
        ## Use back substitution algo to get individual values of beta : L*beta = Y
        
        beta = np.empty(self.Y.shape,dtype=np.float32)
        beta[0] = GarbledDivision()(self.Y[0], self.L[0][0], self.wire_labels)
        beta[0] = np.array([self.wire_labels[wl[i].replace('C','B')] for i in beta[0]])
        for i in range(1,len(beta)):
            print(i)
            s = [ self.wire_labels['A_0'] ] * 24
            for j in range(i):
                temp_lij = np.array([self.wire_labels[wl[i].replace('C','A')] for i in self.L[i][j]])
                mul = GarbledMultiplication()(temp_lij, beta[j], self.wire_labels)
                mul = np.array([self.wire_labels[wl[i].replace('C','B')] for i in mul])
                
                s = GarbledAdd()(s, mul, self.wire_labels)
                s = np.array([self.wire_labels[wl[i].replace('C','A')] for i in s])
                
            s = np.array([self.wire_labels[wl[i].replace('A','B')] for i in s])
            yi = np.array([self.wire_labels[wl[i].replace('B','A')] for i in self.Y[i]])
            beta[i] = GarbledDivision()(yi, s, self.wire_labels)
        
        B = self.convert_to_float(beta)
        print(B.shape,B)
        return beta, self.wire_labels
    
    def execute(self):
        self.calculate_A_b()
        self.calculate_L()
        self.calculate_Y()
        beta,wire_labels = self.calculate_beta()
        return beta,wire_labels    

class CSP(object):
    def __init__(self):
        self.public_key, self.private_key = paillier.generate_paillier_keypair()
        self.d = 1
    
    def paillier_decrypt(self,precision=2):
        #c = user.calculate_and_send_to_evaluater()
        self.A_hat_float, self.b_hat_float = np.empty(self.c_i[0].shape, dtype=np.float32), np.empty(self.c_i[1].shape, dtype=np.float32)
        for i in range(self.c_i[0].shape[0]):
            for j in range(self.c_i[0].shape[0]):
                self.A_hat_float[i][j] = self.private_key.decrypt(self.c_i[0][i][j])

        for j in range(c_i[1].shape[0]):
                self.b_hat_float[j] = self.private_key.decrypt(self.c_i[1][j])
            
    def get_garbled_circuit_and_mua_mub(self):
        self.wire_labels = {}
        self.wire_labels['A_0'] = token_hex(4)
        self.wire_labels['A_1'] = token_hex(4)
        self.wire_labels['B_0'] = token_hex(4)
        self.wire_labels['B_1'] = token_hex(4)
        self.wire_labels['C_0'] = token_hex(4)
        self.wire_labels['C_1'] = token_hex(4)
        
        #self.c_i = self.sock.receive()
        
        #self.A_hat_f, self.b_hat_f = self.paillier_decrypt(self.c_i)
        mu_Af, mu_bf = np.random.randn(self.d,self.d), np.random.randn(self.d)
        
        mu_A_binary = np.empty((self.d,self.d), dtype = np.object)
        mu_b_binary = np.empty((self.d,), dtype = np.object)
        #self.mu_A, self.mu_b = np.empty(self.mu_Af.shape,dtype=np.object), np.empty(self.mu_bf.shape,dtype=np.object)
        
        for i in range(self.d):
            for j in range(self.d):
                mu_A_binary[i][j] = float_to_bin(mu_Af[i][j])
        
        for i in range(self.d):
            mu_b_binary[i] = float_to_bin(mu_bf[i])
        
        mu_A_garbled = np.empty((self.d, self.d, 24,),dtype = np.object)
        mu_b_garbled = np.empty((self.d, self.d, 24,),dtype = np.object)
                    
        for i in range(self.A_hat_b.shape[0]):
            for j in range(self.A_hat_b.shape[1]):
                mu_A_garbled[i][j] = np.array([(self.wire_labels['B_0'] if k=='0' else self.wire_labels['B_1']) for k in mu_A_binary[i][j]])
        
        for i in range(self.b_hat_b.shape[0]):
            mu_b_garbled[i] = np.array([(self.wire_labels['B_0'] if k=='0' else self.wire_labels['B_1']) for k in mu_b_binary[i]])
        
        
        garbled_circuit = GarbledCircuit()
        garbled_circuit.mu_A_garbled = mu_A_garbled
        garbled_circuit.mu_b_garbled = mu_b_garbled
        garbled_circuit.wire_labels = self.wire_labels
        garbled_circuit.public_key = public_key
        #return self.send_to_evaluator(A_hat_garbled,b_hat_garbled,mu_A_garbled,mu_b_garbled,wire_labels,mu_Af,mu_bf)
        return {
            'GARBLED_CIRCUIT': garbled_circuit
            'MU_A_GARBLED' : mu_A_garbled,
            'MU_B_GARBLED' : mu_b_garbled,
            'WIRE_LABELS'  : self.wire_labels,
            'PUBLIC KEY'   : self.public_key,
            'MU_AF'        : mu_Af,
            'MU_BF'        : mu_bf
         }
        
    def get_garbled_a_hat_b_hat(self):
        A_hat_binary = np.empty((self.d,self.d), dtype = np.object)
        b_hat_binary = np.empty((self.d,), dtype = np.object)
        for i in range(self.d):
            for j in range(self.d):
                A_hat_binary[i][j] = float_to_bin(self.A_hat_float[i][j])
        
        for i in range(self.d):
            b_hat_binary[i] = float_to_bin(self.b_hat_float[i])

        A_hat_garbled = np.empty((self.d,self.d, 24,),dtype = np.object)
        b_hat_garbled = np.empty((self.d, self.d, 24,),dtype = np.object)

        for i in range(self.A_hat_b.shape[0]):
            for j in range(self.A_hat_b.shape[1]):
                A_hat_garbled[i][j] = np.array([(self.wire_labels['A_0'] if k=='0' else self.wire_labels['A_1']) for k in A_hat_binary[i][j]])
        
        for i in range(self.b_hat_b.shape[0]):
            b_hat_garbled[i] = np.array([(self.wire_labels['A_0'] if k=='0' else self.wire_labels['A_1']) for k in b_hat_binary[i]])

        return {
            'A_HAT_GARBLED':A_hat_garbled,
            'B_HAT_GARBLED': b_hat_garbled,
        }
if __name__ == '__main__':
    s = socket.socket()     
    csp = CSP()    
    #host = socket.gethostname() # Get local machine name
    host = 'localhost'
    port = 12345                # Reserve a port for your service.
    ENC_MSGS_RECEIVED_FROM_USERS = 'False'
    #print(host)
    s.connect((host, port))
    s.send('CSP: Please send dimensions')
    features = int(s.recv(1024).decode())
    csp.d = features
    s.send('CSP: Sending Public Key, Garbled Circuit & MUs. Please confirm.')
    msg = s.recv(1024).decode()
    if msg == 'Please Send':
        print('Prapring Public/Private Key, Garbled Circuit, Garbled MuA,Mub and Wire Labels Started')
        start = time.time()
	    data = csp.get_garbled_circuit_and_mua_mub()
	    print('Prapring Public/Private Key, Garbled Circuit, Garbled MuA,Mub and Wire Labels Completed')
	    print('Total time taken %s'%(time.time()-start))
    s.send(pickle.dumps(data))
    
    while ENC_MSGS_RECEIVED_FROM_USERS != 'True':
        time.sleep(100)
        s.send('CSP: Please Confirm Whether C Got Calculated')
        s.recv(1024).decode()
    c.send('CSP: Please Send C')
    c_final = pickle.loads(s.recv(1024))
    csp.c_i = c_final
    print('Decrypting C to get A_hat and b_hat Started')
    start = time.time()
    csp.paillier_decrypt()
    print('Decrypting C to get A_hat and b_hat Completed')
    print('Total time taken %s'%(time.time()-start))
    
    print('Calculation of A_hat and b_hat Started')
    start = time.time()
    a_hat_b_hat_garbled = csp.get_garbled_a_hat_b_hat()
    print('Calculation of A_hat and b_hat Completed')
    print('Total time taken %s'%(time.time()-start))
    
    
    s.send('CSP: A_HAT and B_HAT Got Calculated')
    s.send(pickle.dumps(a_hat_b_hat_garbled))            
    print('CSP done with processing. Closing connection')
    s.close()
