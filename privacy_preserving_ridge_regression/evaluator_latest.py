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
                    
        for i in range(self.A_hat_b.shape[0]):
            for j in range(self.A_hat_b.shape[1]):
                mu_A_garbled[i][j] = np.array([(self.wire_labels['B_0'] if k=='0' else self.wire_labels['B_1']) for k in mu_A_binary[i][j]])
        
        for i in range(self.b_hat_b.shape[0]):
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
	num_of_features = 7 ### Please change it according to columns of training set.....................................................................
	evaluator.d = num_of_features
    ENC_MSGS_RECEIVED_FROM_USERS = True
    c_final = None

    while True:
	    c, addr = s.accept()    # Establish connection with client
	    print('Got connection from', addr)
	    msg = c.recv(1024).decode()
	    print(msg)
	    if msg.split(":")[0] == 'CSP':
		    if 'Please send dimensions' in msg:
		        c.send(num_of_features)
		    elif 'Sending Public Key, Garbled Circuit' in msg:
			    c.send('Please Send')
			    gc_pub_key_mua_mub_wires = pickle.loads(c.recv(1024))
#			    evaluator.mu_Af = gc_pub_key_mua_mub_wires['MU_AF']
#    		    evaluator.mu_bf = gc_pub_key_mua_mub_wires['MU_BF']
                garbled_circuit = gc_pub_key_mua_mub_wires['GARBLED_CIRCUIT']
				x = gc_pub_key_mua_mub_wires['RANDOM']
				pub_key_rsa = gc_pub_key_mua_mub_wires['PUBLIC KEY RSA']
				pub_key_paillier = gc_pub_key_mua_mub_wires['PUBLIC KEY PAILLIER']
				evaluator.wire_labels = gc_pub_key_mua_mub_wires['WIRE_LABELS']
				## Received public key of RSA and Random X values. will be doing Oblivious transfer now to get MuAs.
				k,b = int(np.random.choice(100)), int(np.random.choice(5))   ## Randomly generated k and b. (b is index within 0-4 that MuA and Mub will be retrieved)
				v = (x[b] + pow(k,pub_key_rsa.e, pub_key_rsa.n))  ## Generating V
				c.send('Please send MuPrimes. We are sending V. Please confirm.')
				## Next message will send MUPrimes which will be used to get MuA and Mub
			elif 'Please send V' in msg:
				## Sends V of oblivious transfer
				c.send(v)
			elif 'Sending MU Primes. Please confirm' in msg:
				c.send('Confirmed. Please send MUPrimes')

				muprimes = pickle.loads(c.recv(1024))
				evaluator.mu_Af = muprimes['MUA'][b] - k)
				evaluator.mu_bf = muprimes['MUB'][b] - k)
				garbled_circuit.mu_A_garbled, garbled_circuit.mu_b_garbled = evaluator.generate_mu_garbled()

				print('Received MuA and Mub')
            elif 'Please Confirm Whether C Got Calculated' in msg:
                c.send(str(ENC_MSGS_RECEIVED_FROM_USERS))
            elif 'Please Send C' in msg:
				c.send(pickle.dumps(c_final))
            elif 'A_HAT and B_HAT Got Calculated' in msg:
                get_garbled_a_hat_b_hat = c.recv(1024)
                garbled_circuit.A_hat_garbled = get_garbled_a_hat_b_hat['A_HAT_GARBLED']
                garbled_circuit.b_hat_garbled = get_garbled_a_hat_b_hat['B_HAT_GARBLED']
                
                print('Calculating beta Started')
	            start = time.time()
                beta,wire_labels = garbled_circuit.execute()
                print('Calculating beta Completed')
        	    print('Total time taken %s'%(time.time()-start))
        	    
                print(beta,wire_labels)
	    elif msg.split(":")[0] == 'User':
		    if 'Please Send Public Key' in msg:
		        if gc_pub_key_mua_mub_wires:
    	    	    c.send(gc_pub_key_mua_mub_wires['PUBLIC KEY PAILLIER'])
    	    	else:
    	    	    c.send('Please try again after sometime')
	        elif 'Sending Encrypted Messages' in msg:
	            c_i = c.recv(1024)
	            evaluator.c.append(pickle.loads(c_i))
	            if len(evaluator.c) == 2:
	                ENC_MSGS_RECEIVED_FROM_USERS = True
	                print('Calculating C with MuA & Mub Started')
	                start = time.time()
	                c_final = evaluator.calculate_c_hat()
	                print('Calculating C with MuA & Mub Completed')
            	    print('Total time taken %s'%(time.time()-start))
        	    
		            c.send('Thank you for connecting')
			   	    c.close()
