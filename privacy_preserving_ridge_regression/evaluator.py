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
        pass
    
    def calculate_c_hat(self):
        As,bs = zip(*self.c)
        c_final = sum(As),sum(bs)
        c_final = c_final[0]+self.mu_Af,c_final[1]+self.mu_bf
        return c_final
    

if __name__ == '__main__':
    s = socket.socket()
    evaluator = Evaluator()         # Create a socket object
    host = 'localhost'
    port = 12345                # Reserve a port for your service.
    s.bind((host, port))        # Bind to the port
    s.listen(5)                 # Now wait for client connection.
    gc_pub_key_mua_mub_wires = None
    garbled_circuit = None
    ENC_MSGS_RECEIVED_FROM_USERS = True
    c_final = None
    while True:
	    c, addr = s.accept()    # Establish connection with client
	    print('Got connection from', addr)
	    msg = c.recv(1024).decode()
	    print(msg)
	    if msg.split(":")[0] == 'CSP':
		    if 'Please send dimensions' in msg:
		        num_of_features = 7 ## Please change it according to columns of training set
		        c.send(num_of_features)
		    elif 'Sending Public Key, Garbled Circuit & Mus' in msg:
			    c.send('Please Send')
			    gc_pub_key_mua_mub_wires = pickle.loads(c.recv(1024))
			    evaluator.mu_Af = gc_pub_key_mua_mub_wires['MU_AF']
    		    evaluator.mu_bf = gc_pub_key_mua_mub_wires['MU_BF']
                garbled_circuit = gc_pub_key_mua_mub_wires['GARBLED_CIRCUIT']
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
    	    	    c.send(gc_pub_key_mua_mub_wires['key'])
    	    	else:
    	    	    c.send('Please try again after sometime')
	        elif 'Sending Encrypted Messages' in msg:
	            ENC_MSGS_RECEIVED_FROM_USERS = True
	            c_i = c.recv(1024)
	            evaluator.c = pickle.loads(c_i)
	            print('Calculating C with MuA & Mub Started')
	            start = time.time()
	            c_final = evaluator.calculate_c_hat()
	            print('Calculating C with MuA & Mub Completed')
        	    print('Total time taken %s'%(time.time()-start))
        	    
	            c.send('Thank you for connecting')
        	    c.close()
