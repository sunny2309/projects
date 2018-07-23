import pickle
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
import time


class GarbledCircuit(object):
    def __init__(self):
        pass

    def And(self,a,b):
        return a and b

    def Or(self,a,b):
        return a or b

    def Xor(self,a,b):
        return 0 if (a==1 and b==1) or (a==0 and b==0) else 1

    def Not(self,a):
        t = {0:1,1:0}
        return t[a]

    def add(self,a,b,carry):
        #xor = Xor()
        #and_g = And()
        #or_g = Or()
        o1 = self.Xor(a,b)
        o2 = self.And(a,b)
        add = self.Xor(carry,o1)
        o4 = self.And(carry,o1)
        cary = self.Or(o4,o2)
        return str(add), cary
    
    def Add(self,a,b):
        carry = 0
        out = ''
        for idx,i in reversed(list(enumerate(zip(a[:16],b[:16])))):
            o,carry = self.add(int(i[0]),int(i[1]),carry)
            out += o
        #print(out)
        out = ''.join(reversed(out))
        return out + a[16:]
    
    def compute_twos_complement(self,binary):
        out = ''
        for i in binary:
            out += '1' if i=='0' else '0'
        #print(out)
        return Add(out,((len(binary)-1)*'0')+'1')

    def bin_to_float(self,binary):
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

    def int_to_bin(self,num):
        twos_complement = True if num<0 else False
        int_b = bin(int(num))[2:] if num > 0 else bin(int(num))[3:]
        int_b = int_b.zfill(16)
        if twos_complement:
            int_b = compute_twos_complement(int_b)
        return int_b
    
    def bin_to_int_garbled(self,binary,wire_labels):
        wl = dict(zip(wire_labels.values(),wire_labels.keys()))
        b2 = ''.join([wl[i].split('_')[-1] for i in binary])
        int_part = 0
        for i,j in enumerate(reversed(b2[1:])):
            int_part += (int(j) * pow(2, i))
        int_part -= int(wl[binary[0]].split('_')[-1])*pow(2,len(binary)-1)
        return int_part
        
    def replaceLabels(self, l, first, second):
        wl = self.inverseDict()
        return [ self.wire_labels[wl[i].replace(first, second)] for i in l]
    
    def inverseDict(self):
        return dict(zip(self.wire_labels.values(), self.wire_labels.keys()))

    def compute_twos_complement_garbled(self,binary,wire_labels={}):
        wl = self.inverseDict()
        out = []
        for i in binary:
            out += [wire_labels['B_1']] if '0' in wl[i] else [wire_labels['B_0']]
        #print(out)
        out = self.GarbledAdd(((len(binary)-1)*[wire_labels['A_0']])+[wire_labels['A_1']], out, wire_labels)
        out = [wire_labels[wl[i].replace('C','B')] for i in out]
        return out

    def GarbledLeftShift(self,a, n, wire_labels={}, garbled_table=[]):
        a = list(a)
        #print(a,wire_labels)
        wl = self.inverseDict()
        val =  [j for j in [wl[i] for i in list(set(a))] if '0' in j][0]
        return a[n:] + [wire_labels[val]] * n

    def GarbledRightShift(self,a, n, wire_labels={}, garbled_table=[]):
        a = list(a)
        wl = self.inverseDict()
        val =  [j for j in [wl[i] for i in list(set(a))] if '0' in j][0]
        return [wire_labels[val]] * n + a[:-n]

    def GarbledGreaterThanEqualTo(self,a, b, wire_labels):
        return self.bin_to_int_garbled(a, wire_labels) >= self.bin_to_int_garbled(b, wire_labels)
    
    def GarbledPadZerosLeft(self,a, n, wire_labels={}, garbled_table=[]):
        wl = self.inverseDict()
        val =  [j for j in [wl[i] for i in list(set(a))] if '0' in j][0]
        return [wire_labels[val]] * n + a 

    def GarbledPadZerosRight(self,a, n, wire_labels={}, garbled_table=[]):
        wl = self.inverseDict()
        val =  [j for j in [wl[i] for i in list(set(a))] if '0' in j][0]
        return a + [wire_labels[val]] * n
    
    def GarbledAnd(self,a,b,wire_labels={},garbled_table=[]):
        
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
        
    def GarbledOr(self,a,b,wire_labels={},garbled_table=[]):
        
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

    def GarbledXor(self,a,b,wire_labels={},garbled_table=[]):
        
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
        
    def GarbledNot(self,b,wire_labels={},garbled_table=[]):
        
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
            
    def add(self,a,b,carry,wire_labels):
        wl = self.inverseDict()
        
        des_in = DES.new(b, DES.MODE_ECB)
        des_out = DES.new(a, DES.MODE_ECB)
        
        o1 = self.GarbledXor(a,b,wire_labels)
        #print(a,b,o1)
        o1 = des_in.decrypt(des_out.decrypt(o1)).decode()
        
        o2 = self.GarbledAnd(a,b,wire_labels)
        o2 = des_in.decrypt(des_out.decrypt(o2)).decode()
        
        o1 = wire_labels[wl[o1].replace('C','B')]
        
        des_in = DES.new(o1, DES.MODE_ECB)
        des_out = DES.new(carry, DES.MODE_ECB)
        
        add = self.GarbledXor(carry,o1,wire_labels)
        add = des_in.decrypt(des_out.decrypt(add)).decode()
        
        o4 = self.GarbledAnd(carry,o1,wire_labels)
        o4 = des_in.decrypt(des_out.decrypt(o4)).decode()
        
        o4 = wire_labels[wl[o4].replace('C', 'A')]
        o2 = wire_labels[wl[o2].replace('C', 'B')]
        
        des_in = DES.new(o2, DES.MODE_ECB)
        des_out = DES.new(o4, DES.MODE_ECB)
        
        cary = self.GarbledOr(o4, o2, wire_labels)
        cary = des_in.decrypt(des_out.decrypt(cary)).decode()
        
        cary = wire_labels[wl[cary].replace('C', 'A')]
        return str(add), cary
    
    def GarbledAdd(self,a, b, wire_labels={}):
        wl = self.inverseDict()
        carry = wire_labels['A_0']
        out = []
        for idx,i in reversed(list(enumerate(zip(a[:16],b[:16])))):
            o,carry = self.add(i[0], i[1], carry, wire_labels)
            out.append(o)
        #print(out)
        out = list(reversed(out))
        return out + self.replaceLabels(a[16:], 'A', 'C')

    def GarbledSubtract(self,a,b, wire_labels={}):
        b2 = self.compute_twos_complement_garbled(b[:16],wire_labels)
        #print(b2,b[16:])
        b2 = list(b2) + list(b[16:])
        return self.GarbledAdd(a,b2,wire_labels)

    def GarbledMultiplication(self,a,b, wire_labels={}):
        a,b =list(a),list(b)
        wl = self.inverseDict()
        negative_result = False
        if self.bin_to_int_garbled(a[:16],wire_labels) < 0 and self.bin_to_int_garbled(b[:16],wire_labels) < 0:
            temp = self.compute_twos_complement_garbled(a[:16],wire_labels)
            temp = self.replaceLabels(temp,'B','A')
            a = temp + a[16:]
            b = self.compute_twos_complement_garbled(b[:16], wire_labels) + b[16:]
        elif self.bin_to_int_garbled(a[:16], wire_labels) < 0:
            temp = self.compute_twos_complement_garbled(a[:16], wire_labels)
            temp = self.replaceLabels(temp,'B','A')
            a = temp + a[16:]
            negative_result = True
        elif self.bin_to_int_garbled(b[:16],wire_labels) < 0:
            b = self.compute_twos_complement_garbled(b[:16],wire_labels) + b[16:]
            negative_result = True

        s = [wire_labels['B_0']]*16
        m = copy.copy(a[:16])
        for i in reversed(b[:16]):
            if '1' in wl[i]:     # when digit is one, add the intermediary product
                s = self.GarbledAdd(m, s, wire_labels)
                s = self.replaceLabels(s,'C','B')
            m = self.GarbledLeftShift(m, 1, wire_labels)  # shift one per digit in b
        s = self.replaceLabels(s,'B','C')
        s = self.GarbledRightShift(s,4,wire_labels) ## Shifting left by 4 places to get exponent back to 2^-4
        
        if negative_result:
            s = self.replaceLabels(s,'C','B')
            s = self.compute_twos_complement_garbled(s, wire_labels)
            s = self.replaceLabels(s,'B','C')
            
        return s+self.replaceLabels(b[:16],'B','C')
        
    def GarbledDivision(self,a, b,wire_labels={}):
        a,b = list(a),list(b)
        #print(self.bin_to_int_garbled(a[:16],wire_labels),self.bin_to_int_garbled(b[:16],wire_labels))
        wl = self.inverseDict()
        negative_result = False
        if self.bin_to_int_garbled(a[:16],wire_labels) < 0 and self.bin_to_int_garbled(b[:16],wire_labels) < 0:
            temp = self.compute_twos_complement_garbled(a[:16],wire_labels)
            temp = self.replaceLabels(temp,'B','A')
            a = temp + a[16:]
            b = self.compute_twos_complement_garbled(b[:16], wire_labels) + b[16:]
        elif self.bin_to_int_garbled(a[:16], wire_labels) < 0:
            temp = self.compute_twos_complement_garbled(a[:16], wire_labels)
            temp = self.replaceLabels(temp,'B','A')
            a = temp + a[16:]
            negative_result = True
        elif self.bin_to_int_garbled(b[:16],wire_labels) < 0:
            b = self.compute_twos_complement_garbled(b[:16],wire_labels) + b[16:]
            negative_result = True
        exp = 0
        if self.bin_to_int_garbled(a[:16],wire_labels) >= self.bin_to_int_garbled(b[:16],wire_labels):
            reminder = self.GarbledSubtract(a[:16], b[:16],wire_labels)
            reminder = self.replaceLabels(reminder,'C','A')
            quotient = self.GarbledAdd([ wire_labels['A_0'] ]*16, (15*[ wire_labels['B_0'] ])+ [ wire_labels['B_1'] ],wire_labels)
            quotient = self.replaceLabels(quotient,'C','A')
            while self.bin_to_int_garbled(reminder, wire_labels) > self.bin_to_int_garbled(b[:16], wire_labels):
                reminder = self.GarbledSubtract(reminder, b[:16], wire_labels)
                reminder = self.replaceLabels(reminder,'C','A')
                quotient = self.GarbledAdd(quotient, (15*[ wire_labels['B_0'] ])+ [ wire_labels['B_1'] ], wire_labels)
                quotient = self.replaceLabels(quotient,'C','A')
            for i in range(2):
                if self.bin_to_int_garbled(reminder, wire_labels) == 0:
                    break
                while self.bin_to_int_garbled(reminder, wire_labels) < self.bin_to_int_garbled(b[:16], wire_labels) and \
                    self.bin_to_int_garbled(reminder,wire_labels) != 0:
                    reminder = self.GarbledLeftShift(reminder,1, wire_labels)
                    quotient = self.GarbledLeftShift(quotient, 1, wire_labels)
                    exp -= 1
                reminder = self.GarbledSubtract(reminder, b[:16], wire_labels)
                reminder = self.replaceLabels(reminder,'C','A')
                quotient = self.GarbledAdd(quotient, (15*[ wire_labels['B_0'] ])+ [ wire_labels['B_1'] ], wire_labels)
                quotient = self.replaceLabels(quotient,'C','A')
            quotient = self.replaceLabels(quotient,'A','C')
        else:
            reminder = list(copy.copy(a[:16]))
            exp = -1
            reminder = self.GarbledLeftShift(reminder,1, wire_labels)
            quotient = [ wire_labels['A_0'] ]*16
            for i in range(2):
                if self.bin_to_int_garbled(reminder, wire_labels) == 0:
                    break
                while self.bin_to_int_garbled(reminder,wire_labels) < self.bin_to_int_garbled(b[:16], wire_labels) and \
                    self.bin_to_int_garbled(reminder,wire_labels) !=0:
                    reminder = self.GarbledLeftShift(reminder,1, wire_labels)
                    quotient = self.GarbledLeftShift(quotient,1, wire_labels)
                    exp -= 1
            reminder = self.GarbledSubtract(reminder, b[:16],wire_labels)
            quotient = self.GarbledAdd(quotient, (15*[ wire_labels['B_0'] ])+ [ wire_labels['B_1'] ], wire_labels)
        quotient = self.GarbledRightShift(quotient, abs(exp) - 4, wire_labels) if exp < -4 else self.GarbledLeftShift(quotient,4 - abs(exp), wire_labels)
        if negative_result:
            quotient = self.replaceLabels(quotient,'C','B')
            quotient = self.compute_twos_complement_garbled(quotient, wire_labels)
            quotient = self.replaceLabels(quotient,'B','C')
        return quotient+self.replaceLabels(a[16:],'A','C')
    
    def GarbledSqrt(self,x,wire_labels={}):
        wl = self.inverseDict()
        num = copy.copy(x[:16])
        xx = self.bin_to_int_garbled(x[:16],wire_labels)
        e = [i for i in range(int(xx)) if pow(4,i)<=xx]
        e = self.int_to_bin(pow(4,e[-1])) if e else '0'*len(num)
        e = [wire_labels['B_'+str(i)] for i in e]
        if len(e) < len(num):
            e = self.GarbledPadZerosLeft(e,len(num) - len(e),wire_labels)
        r = [ wire_labels['A_0'] ]*len(e)
        while wire_labels['B_1'] in e:
            if self.GarbledGreaterThanEqualTo(num, self.GarbledAdd(r, e, wire_labels), wire_labels):
                out = self.GarbledAdd(r, e, wire_labels)
                out = self.replaceLabels(out,'C','B')
                num = self.GarbledSubtract(num, out,wire_labels)
                num = self.replaceLabels(num,'C','A')
                r = self.GarbledAdd(self.GarbledRightShift(r, 1, wire_labels), e, wire_labels)
                r = self.replaceLabels(r,'C','A')
            else:
                r = self.GarbledRightShift(r, 1, wire_labels)
            e = self.GarbledRightShift(e, 2, wire_labels)
        r = self.GarbledLeftShift(r,2, wire_labels)
        return r + self.replaceLabels(x[16:],'A','C')

    def convert_to_float(self,a,two_dimension=True):
        out = np.empty(a.shape[:-1],dtype=np.float)
        wl = self.inverseDict()
        if two_dimension:
            for i in range(a.shape[0]):
                for j in range(a.shape[1]):
                    out[i][j] = self.bin_to_float(''.join([ wl[k].split('_')[-1] for k in a[i][j] ]))
        else:
            for i in range(a.shape[0]):
                out[i] = self.bin_to_float(''.join([ wl[k].split('_')[-1] for k in a[i] ]))
                
        return out
    
    def calculate_A_b(self):
        ## Subtract mu_A and mu_B from A_hat and b_hat
        self.A_garbled = np.empty(self.A_hat_garbled.shape,dtype=np.object)
        self.b_garbled = np.empty(self.b_hat_garbled.shape,dtype=np.object)
        for i in range(self.A_hat_garbled.shape[0]):
            for j in range(self.A_hat_garbled.shape[1]):
                self.A_garbled[i][j] = np.array(self.GarbledSubtract(self.A_hat_garbled[i][j],self.mu_A_garbled[i][j],self.wire_labels))
        
        for i in range(self.b_hat_garbled.shape[0]):
            self.b_garbled[i] = np.array(self.GarbledSubtract(self.b_hat_garbled[i],self.mu_b_garbled[i], self.wire_labels))
        
        A = self.convert_to_float(self.A_garbled)
        b = self.convert_to_float(self.b_garbled,False)
        print('A',A.shape,A)
        print('B',b.shape,b)
        
    def calculate_L(self):
        ## Get L using A with algo cholesky
        wl = self.inverseDict()
        self.d = self.A_garbled.shape[0]
        self.L = self.A_garbled.copy()
        for i in range(self.d):
            for j in range(self.d):
                self.L[i][j] = np.array(self.replaceLabels(self.L[i][j],'C','A'))
        for i in range(self.d-1):
            for j in range(i+1,self.d):
                self.L[i][j] = np.array([ self.wire_labels['A_0'] ]*24)
        for j in range(self.d):
            for k in range(j):
                for i in range(j,self.d):
                    ljk = self.replaceLabels(self.L[j][k],'A','B')
                    mul = self.GarbledMultiplication(self.L[i][k], ljk, self.wire_labels)
                    mul = self.replaceLabels(mul,'C','B')
                    self.L[i][j] = np.array(self.GarbledSubtract(self.L[i][j], mul, self.wire_labels)) ## Need to bring -4 as exponent
                    self.L[i][j] = np.array(self.replaceLabels(self.L[i][j],'C','A'))
            
            self.L[j][j] = np.array(self.GarbledSqrt(self.L[j][j],self.wire_labels)) ## Need to bring -4 as exponent
            self.L[j][j] = np.array(self.replaceLabels(self.L[j][j],'C','A'))
            
            for k in range(j+1,self.d):
                ljj = self.replaceLabels(self.L[j][j],'A','B')
                self.L[k][j] = np.array(self.GarbledDivision(self.L[k][j], ljj, self.wire_labels)) ## Need to bring -4 as exponent
                self.L[k][j] = np.array(self.replaceLabels(self.L[k][j],'C','A'))
        
        LF = self.convert_to_float(self.L)
        print('L', LF.shape,LF)
        
    def calculate_Y(self):
        ## Use back substitution algo to get individual values of Y    : L.T * Y = b
        wl = self.inverseDict()
        LT = self.L.transpose(1,0,2)
        self.Y  = np.empty(self.b_garbled.shape,dtype=np.object)
        b_d_1 = self.replaceLabels(self.b_garbled[self.d-1],'C','A')
        lt_d_1 = self.replaceLabels(LT[self.d-1][self.d-1],'A','B')
        self.Y[self.d-1] = np.array(self.GarbledDivision(b_d_1, lt_d_1, self.wire_labels))
        self.Y[self.d-1] = np.array(self.replaceLabels(self.Y[self.d-1],'C','B'))
        for i in reversed(range(self.d-1)):
            s = [ self.wire_labels['A_0'] ] * 24
            for j in range(i+1,self.d):
                mul = self.GarbledMultiplication(LT[i][j], self.Y[j], self.wire_labels)
                mul = self.replaceLabels(mul,'C','B')
                
                s = self.GarbledAdd(s, mul, self.wire_labels)
                s = self.replaceLabels(s,'C','A')
            
            bi = self.replaceLabels(self.b_garbled[i],'C','A')
            s = self.replaceLabels(s,'A','B')
            s = self.GarbledSubtract(bi, s, self.wire_labels)
            s = self.replaceLabels(s,'C','A')
            lt = self.replaceLabels(LT[i][i],'A','B')
            self.Y[i] = np.array(self.GarbledDivision(s, lt, self.wire_labels))
            self.Y[i] = np.array(self.replaceLabels(self.Y[i],'C','B'))
        
        YF = self.convert_to_float(self.Y,False)
        print('Y', YF.shape,YF)
        
    def calculate_beta(self):
        ## Use back substitution algo to get individual values of beta : L*beta = Y
        wl = self.inverseDict()
        beta = np.empty(self.Y.shape,dtype=np.object)
        y0 = self.replaceLabels(self.Y[0],'B','A')
        l00 = self.replaceLabels(self.L[0][0],'A','B')
        beta[0] = np.array(self.GarbledDivision(y0, l00, self.wire_labels))
        beta[0] = np.array(self.replaceLabels(beta[0],'C','B'))
        for i in range(1,len(beta)):
            s = [ self.wire_labels['A_0'] ] * 24
            for j in range(i):
                mul = self.GarbledMultiplication(self.L[i][j], beta[j], self.wire_labels)
                mul = self.replaceLabels(mul,'C','B')
                
                s = self.GarbledAdd(s, mul, self.wire_labels)
                s = self.replaceLabels(s,'C','A')
            
            yi = self.replaceLabels(self.Y[i],'B','A')
            s = self.replaceLabels(s,'A','B')
            s = self.GarbledSubtract(yi, s, self.wire_labels)
            s = self.replaceLabels(s,'C','A')
            l = self.replaceLabels(self.L[i][i],'A','B')
            
            beta[i] = np.array(self.GarbledDivision(s, l, self.wire_labels))
            beta[i] = np.array(self.replaceLabels(beta[i],'C','B'))
        
        B = self.convert_to_float(beta,False)
        print('B', B.shape,B)
        #return beta, self.wire_labels
    
    def execute(self):
        self.calculate_A_b()
        self.calculate_L()
        self.calculate_Y()
        self.calculate_beta()
        #return B, beta,wire_labels
