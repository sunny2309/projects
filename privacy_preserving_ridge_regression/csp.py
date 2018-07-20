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
import rsa
from garbled_circuit import GarbledCircuit

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
    
        
class CSP(object):
    def __init__(self):
        self.public_key, self.private_key = paillier.generate_paillier_keypair()
        self.d = 1
        self.pub,self.priv = rsa.newkeys(32)
        self.x = np.random.choice(100,5)
    
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
        
        #mu_Af, mu_bf = np.random.randn(self.d,self.d), np.random.randn(self.d)
        
        garbled_circuit = GarbledCircuit()
        #garbled_circuit.mu_A_garbled = mu_A_garbled
        #garbled_circuit.mu_b_garbled = mu_b_garbled
        garbled_circuit.wire_labels = self.wire_labels
        garbled_circuit.public_key = self.public_key
        #return self.send_to_evaluator(A_hat_garbled,b_hat_garbled,mu_A_garbled,mu_b_garbled,wire_labels,mu_Af,mu_bf)
        return {
            'GARBLED_CIRCUIT': garbled_circuit,
            #'MU_A_GARBLED' : mu_A_garbled,
            #'MU_B_GARBLED' : mu_b_garbled,
            'WIRE_LABELS'  : self.wire_labels,
            'PUBLIC KEY PAILLIER'   : self.public_key,
			'PUBLIC KEY RSA' : self.pub,
			'RANDOM' : self.x
            #'MU_AF'        : mu_Af,
            #'MU_BF'        : mu_bf
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
    data, muprimes = None, None
    #host = socket.gethostname() # Get local machine name
    host = 'localhost'
    port = 12345                # Reserve a port for your service.
    ENC_MSGS_RECEIVED_FROM_USERS = 'False'
    #print(host)
    s.connect((host, port))
    s.send(pickle.dumps('CSP: Please send dimensions'))
    features = pickle.loads(s.recv(1024))
    print(features)
#    features = int()
    csp.d = int(features)
    csp.muAs = [np.random.randn(csp.d,csp.d) for i in range(5)]
    csp.muBs = [np.random.randn(csp.d) for i in range(5)]
    print('Test')
    s.send(pickle.dumps('CSP: Sending Public Key, Garbled Circuit & MUs. Please confirm.'))
    print('Test')
    msg = pickle.loads(s.recv(1024))
    print(msg)
    if msg == 'Please Send':
        data = csp.get_garbled_circuit_and_mua_mub()
    s.send(pickle.dumps(data))
    msg = pickle.loads(s.recv(1024))
    print(msg)
    if 'Please send MuPrimes' in msg:
        ## Asking for V which will be used to compute MuPrimes for oblivious transfer
        s.send(pickle.dumps('CSP: Please send V'))
        v = int(pickle.loads(s.recv(1024)))
        print('V : '+str(v))
        ks = [(pow(int((v-i)),csp.priv.d,csp.priv.n))  for i in csp.x]
        ## Calculated Mu primes which will be sent to Evaluator from which evaluator will get Mus for which he asked.
        muA_primes = [muA+ki for muA,ki in zip(csp.muAs,ks)]  
        mub_primes = [mub+ki for mub,ki in zip(csp.muBs,ks)]
        muprimes = {'MUA':muA_primes, 'MUB':mub_primes}
        print('Reached Here')
        s.send(pickle.dumps('CSP: Sending MU Primes. Please confirm'))

    msg = pickle.loads(s.recv(1024))
    print(msg)
    if 'Confirmed. Please send MUPrimes' in msg:
        s.send(pickle.dumps(muprimes))

    while ENC_MSGS_RECEIVED_FROM_USERS != 'True':
        time.sleep(100)
        s.send(pickle.dumps('CSP: Please Confirm Whether C Got Calculated'))
        ENC_MSGS_RECEIVED_FROM_USERS =  pickle.loads(s.recv(1024))
    c.send(pickle.dumps('CSP: Please Send C'))
    c_final = pickle.loads(s.recv(1024))
    csp.c_i = c_final
    csp.paillier_decrypt()
    a_hat_b_hat_garbled = csp.get_garbled_a_hat_b_hat()
    s.send(pickle.dumps('CSP: A_HAT and B_HAT Got Calculated'))
    s.send(pickle.dumps(a_hat_b_hat_garbled))            
    s.close()
