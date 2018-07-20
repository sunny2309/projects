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
    for idx,i in reversed(list(enumerate(zip(a[:16],b[:16])))):
        o,carry = add(int(i[0]),int(i[1]),carry)
        out += o
    #print(out)
    out = ''.join(reversed(out))
    return out + a[16:]


def compute_twos_complement(binary):
    out = ''
    for i in binary:
        out += '1' if i=='0' else '0'
    #print(out)
    return Add(out,((len(binary)-1)*'0')+'1')

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

class Evaluator(object):
    def __init__(self):
        self.c = []
    
    def calculate_c_hat(self):
        As,bs = zip(*self.c)
        c_final = sum(As),sum(bs)
        c_final = c_final[0]+self.mu_Af,c_final[1]+self.mu_bf
        return c_final
    
    def generate_mu_garbled(self):
        mu_A_binary = np.empty((self.d,self.d), dtype = np.object)
        mu_b_binary = np.empty((self.d,), dtype = np.object)
                
        for i in range(self.d):
            for j in range(self.d):
                mu_A_binary[i][j] = float_to_bin(self.mu_Af[i][j])
        
        for i in range(self.d):
            mu_b_binary[i] = float_to_bin(self.mu_bf[i])
        
        mu_A_garbled = np.empty((self.d, self.d, 24,),dtype = np.object)
        mu_b_garbled = np.empty((self.d, self.d, 24,),dtype = np.object)
                    
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
    ENC_MSGS_RECEIVED_FROM_USERS = True
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
	                print(gc_pub_key_mua_mub_wires)
	                garbled_circuit = gc_pub_key_mua_mub_wires['GARBLED_CIRCUIT']
	                x = gc_pub_key_mua_mub_wires['RANDOM']
	                pub_key_rsa = gc_pub_key_mua_mub_wires['PUBLIC KEY RSA']
	                pub_key_paillier = gc_pub_key_mua_mub_wires['PUBLIC KEY PAILLIER']
	                evaluator.wire_labels = gc_pub_key_mua_mub_wires['WIRE_LABELS']
	                ## Received public key of RSA and Random X values. will be doing Oblivious transfer now to get MuAs.
	                k,b = int(np.random.choice(100)), int(np.random.choice(5))   ## Randomly generated k and b. (b is index within 0-4 that MuA and Mub will be retrieved)
	                v = (x[b] + pow(k,pub_key_rsa.e, pub_key_rsa.n))  ## Generating V
	                print('Reached Here')
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
	                    evaluator.mu_Af = muprimes['MUA'][b] - k
	                    evaluator.mu_bf = muprimes['MUB'][b] - k
	                    garbled_circuit.mu_A_garbled, garbled_circuit.mu_b_garbled = evaluator.generate_mu_garbled()
	        elif 'Please Confirm Whether C Got Calculated' in msg:
	            c.send(pickle.dumps(str(ENC_MSGS_RECEIVED_FROM_USERS)))
	        elif 'Please Send C' in msg:
	            c.send(pickle.dumps(c_final))
	        elif 'A_HAT and B_HAT Got Calculated' in msg:
	            get_garbled_a_hat_b_hat = c.recv(1024)
	            garbled_circuit.A_hat_garbled = get_garbled_a_hat_b_hat['A_HAT_GARBLED']
	            garbled_circuit.b_hat_garbled = get_garbled_a_hat_b_hat['B_HAT_GARBLED']
	            beta,wire_labels = garbled_circuit.execute()
	    elif msg.split(":")[0] == 'User':
	        if 'Please Send Public Key' in msg:
	            if gc_pub_key_mua_mub_wires:
	                c.send(pickle.dumps(gc_pub_key_mua_mub_wires['PUBLIC KEY PAILLIER']))
	            else:
	                c.send(pickle.dumps('Please try again after sometime'))
	        elif 'Sending Encrypted Messages' in msg:
	            c_i = c.recv(1024)
	            evaluator.c.append(pickle.loads(c_i))
	            if len(evaluator.c) == 2:
	                ENC_MSGS_RECEIVED_FROM_USERS = True
	                c_final = evaluator.calculate_c_hat()
	                c.send(pickle.dumps('Thank you for connecting'))
	                c.close()
