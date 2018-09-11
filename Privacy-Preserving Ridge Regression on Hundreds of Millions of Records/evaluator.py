## evaluator.py
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
from garbled_circuit import GarbledCircuit

size = 64 ## Total size of fixed point representation
exp = 8 ## 8 bits from above will be used to represent exponent
intp = size - exp ## Remaining 56 bits will be used to represent integer
fracp = 16 ## Number of times loop should run for fractional part in float_t_bin method

def And(a,b):
    return a and b

def Or(a,b):
    return a or b

def Xor(a,b):
    return 0 if (a==1 and b==1) or (a==0 and b==0) else 1

def Not(a):
    t = {0:1,1:0}
    return t[a]

def add(a,b,carry):
    #xor = Xor()
    #and_g = And()
    #or_g = Or()
    o1 = Xor(a,b)
    o2 = And(a,b)
    add = Xor(carry,o1)
    o4 = And(carry,o1)
    cary = Or(o4,o2)
    return str(add), cary

def Add(a,b):
    carry = 0
    out = ''
    for idx,i in reversed(list(enumerate(zip(a[:intp],b[:intp])))):
        o,carry = add(int(i[0]),int(i[1]),carry)
        out += o
    #print(out)
    out = ''.join(reversed(out))
    return out + a[intp:]


def compute_twos_complement(binary):
    out = ''
    for i in binary:
        out += '1' if i=='0' else '0'
    #print(out)
    return Add(out,((len(binary)-1)*'0')+'1')
    
def int_to_bin(num):
    twos_complement = True if num<0 else False
    int_b = bin(int(num))[2:] if num > 0 else bin(int(num))[3:]
    if twos_complement:
        int_b = compute_twos_complement(int_b)
    return int_b

def float_to_bin(num):
    twos_complement = True if num<0 else False
    int_p,frac_p = str(num).split('.')
    frac_p = float('0.'+frac_p)
    int_b = bin(int(int_p))[2:] if num > 0 else bin(int(int_p))[3:]
    frac_b = '.'
    for i in range(fracp):
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
    #out = out[:-4]  ## Need to change to -8 in future
    exponent = compute_twos_complement(int_to_bin(fracp).zfill(exp))
    #print(exponent)
    out += exponent     #'11110000'
    out = out.zfill(size)
    #print(out)
    if twos_complement:
        temp_out = compute_twos_complement(out[:size-exp])
    #print(temp_out)
    out = temp_out+out[size-exp:] if twos_complement else out
    #print(out)
    return out

class Evaluator(object):
    def __init__(self):
        self.c = []
    
    def paillier_encrypt(self, A_i, b_i, precision=2):
        '''
           This method encrypts C_Mus = (MuA,Mub) generated by below method. It iterates through each value of array and \ 
           encrypts it.
        '''
        A_i_enc = np.zeros(A_i.shape,dtype=np.object)
        for i in range(A_i.shape[0]):
            for j in range(A_i.shape[1]):
                #print(A_i[i][j])
                A_i_enc[i][j] = self.public_key.encrypt(A_i[i][j],precision=precision)
                
        b_i_enc = np.zeros(b_i.shape,dtype=np.object)
        for i in range(b_i.shape[0]):
            b_i_enc[i] = self.public_key.encrypt(b_i[i],precision=precision)
        
        return A_i_enc, b_i_enc
    
    def calculate_c_hat(self):
        '''
        This method sums up all C's received from each user. After summing up it adds MuA and Mub to A & b to generate 
        C_hat = (A_hat,b_hat)
        '''
        As,bs = zip(*self.c)
        c_final = sum(As),sum(bs)
        mu_Af_enc,mu_bf_enc = self.paillier_encrypt(self.mu_Af,self.mu_bf)
        c_final = c_final[0]+mu_Af_enc,c_final[1]+mu_bf_enc
        return c_final
    
    def generate_mu_garbled(self):
        mu_A_binary = np.empty((self.d,self.d), dtype = np.object)
        mu_b_binary = np.empty((self.d,), dtype = np.object)
                
        for i in range(self.d):
            for j in range(self.d):
                mu_A_binary[i][j] = float_to_bin(self.mu_Af[i][j])
        
        for i in range(self.d):
            mu_b_binary[i] = float_to_bin(self.mu_bf[i])
        
        mu_A_garbled = np.empty((self.d, self.d, size,),dtype = np.object)
        mu_b_garbled = np.empty((self.d, size,),dtype = np.object)
                    
        for i in range(mu_A_binary.shape[0]):
            for j in range(mu_A_binary.shape[1]):
                mu_A_garbled[i][j] = np.array([(self.wire_labels['B_0'] if k=='0' else self.wire_labels['B_1']) for k in mu_A_binary[i][j]])
        
        for i in range(mu_b_binary.shape[0]):
            mu_b_garbled[i] = np.array([(self.wire_labels['B_0'] if k=='0' else self.wire_labels['B_1']) for k in mu_b_binary[i]])
        
        return mu_A_garbled, mu_b_garbled

if __name__ == '__main__':
    s = socket.socket()
    evaluator = Evaluator()         # Create a socket object
    host = 'localhost'
    port = 12345                # Reserve a port for your service.
    s.bind((host, port))        # Bind to the port
    s.listen(5)                 # Now wait for client connection.
    gc_pub_key_mua_mub_wires, garbled_circuit, x,pub_key_rsa, pub_key_paillier, v, muprimes = None, None, None, None, None, None, None
    k,b,muAf,mubf = None, None, None, None
    num_of_features = 7 ### Please change it according to columns of training set................................................
    evaluator.d = num_of_features
    ENC_MSGS_RECEIVED_FROM_USERS = 'False'
    c_final = None

    while True:
	    c, addr = s.accept()    # Establish connection with client
	    print('Got connection from', addr)
	    msg = pickle.loads(c.recv(1024))
	    print(msg)
	    if msg.split(":")[0] == 'CSP':
	        if 'Please send dimensions' in msg:
	            c.send(pickle.dumps(num_of_features))
	            msg = pickle.loads(c.recv(1024))
	            print(msg)
	            if 'Sending Public Key, Garbled Circuit' in msg:
	                c.send(pickle.dumps('Please Send'))
	                gc_pub_key_mua_mub_wires = pickle.loads(c.recv(4096))
	                #print(gc_pub_key_mua_mub_wires)
	                garbled_circuit = gc_pub_key_mua_mub_wires['GARBLED_CIRCUIT']
	                x = gc_pub_key_mua_mub_wires['RANDOM']
	                pub_key_rsa = gc_pub_key_mua_mub_wires['PUBLIC KEY RSA']
	                pub_key_paillier = gc_pub_key_mua_mub_wires['PUBLIC KEY PAILLIER']
	                evaluator.wire_labels = gc_pub_key_mua_mub_wires['WIRE_LABELS']
	                ## Oblivioous transfer starts from here in evaluator part after it receives random X's from CSP which it uses to ask for MuA and Mub.
	                ## Received public key of RSA and Random X values. will be doing Oblivious transfer now to get MuAs.
	                k,b = int(np.random.choice(100)), int(np.random.choice(5))   ## Randomly generated k and b. (b is index within 0-4 that MuA and Mub will be retrieved)
	                v = (x[b] + pow(k,pub_key_rsa.e, pub_key_rsa.n))  ## Generating V
	                c.send(pickle.dumps('Please send MuPrimes. We are sending V. Please confirm.'))
	                msg = pickle.loads(c.recv(1024))
	                print(msg)
	                if 'Please send V' in msg:
        	            c.send(pickle.dumps(v))
        	        msg = pickle.loads(c.recv(1024))
	                print(msg)
	                if 'Sending MU Primes. Please confirm' in msg:
	                    c.send(pickle.dumps('Confirmed. Please send MUPrimes'))
	                    muprimes = pickle.loads(c.recv(4096))
	                    ## Once we get encrypted MuAs and Mubs from users from it we'll retrieve MuA and Mub using b & k.
	                    evaluator.mu_Af = muprimes['MUA'][b] - k
	                    evaluator.mu_bf = muprimes['MUB'][b] - k
	                    evaluator.public_key = gc_pub_key_mua_mub_wires['PUBLIC KEY PAILLIER']
	                    #print(evaluator.mu_Af,evaluator.mu_bf)
	                    garbled_circuit.mu_A_garbled, garbled_circuit.mu_b_garbled = evaluator.generate_mu_garbled()
	                    
	                    #print('Reached Here')
	                ## Preparation phase has completed now. Now CSP will come back online when C gets calculated from User side.
	                msg = pickle.loads(c.recv(1024))
	                print(msg)
	                if 'Please Confirm Whether C Got Calculated' in msg:
	                    c.send(pickle.dumps(str(ENC_MSGS_RECEIVED_FROM_USERS)))
	        elif 'Please Confirm Whether C Got Calculated' in msg:
	            c.send(pickle.dumps(str(ENC_MSGS_RECEIVED_FROM_USERS)))
	        elif 'Please Send C' in msg:
	            c.send(pickle.dumps(c_final))
	        elif 'A_HAT and B_HAT Got Calculated' in msg:
	            data = b''
	            try:
	                while True:
	                    data_new = c.recv(1000)
	                    if data_new:
	                        data += data_new
	                        c.settimeout(1)
	                    else:
	                        print ("Data null")
	                        break
	            finally:
	                get_garbled_a_hat_b_hat = pickle.loads(data)
	            print(get_garbled_a_hat_b_hat['A_HAT_GARBLED'].shape, get_garbled_a_hat_b_hat['B_HAT_GARBLED'].shape)
	            garbled_circuit.A_hat_garbled = get_garbled_a_hat_b_hat['A_HAT_GARBLED']
	            garbled_circuit.b_hat_garbled = get_garbled_a_hat_b_hat['B_HAT_GARBLED']
	            print('Executing Garbled Circuit to generate beta started')
	            start = time.time()
	            garbled_circuit.execute()
	            print('Executing Garbled Circuit to generate beta completed. Time taken : %.2f seconds'%(time.time() - start))
	            print('Closing Evaluator. Beta got calculated')
	            s.close()
	            break
	    elif msg.split(":")[0] == 'User':
	        if 'Please Send Public Key' in msg:
	            if gc_pub_key_mua_mub_wires:
	                c.send(pickle.dumps(gc_pub_key_mua_mub_wires['PUBLIC KEY PAILLIER']))
	            else:
	                c.send(pickle.dumps('Please try again after sometime'))
	        elif 'Sending Encrypted Messages' in msg:
	            data = b''
	            try:
	                while True:
	                    data_new = c.recv(1024)
	                    if data_new:
	                        data += data_new
	                        c.settimeout(1)
	                    else:
	                        break
	            finally:
	                c_i = pickle.loads(data)
	            evaluator.c.append(c_i)
	            if len(evaluator.c) == 4:     ## Set it equal to number of users
	                ENC_MSGS_RECEIVED_FROM_USERS = 'True'
	                print('Calculating C_hat adding MuA and Mub to C from users started')
	                start = time.time()
	                c_final = evaluator.calculate_c_hat()
	                print('Calculating C_hat adding MuA and Mub to C from users completed. Time taken : %.2f seconds'%(time.time() - start))
	                c.send(pickle.dumps('Thank you for connecting'))
	                c.close()