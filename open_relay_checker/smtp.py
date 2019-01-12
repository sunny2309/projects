
#!/usr/bin/python
from __future__ import print_function
import socket,sys
import smtplib
from smtplib import *
#from Queue import Queue
import time
import threading
import os, threading
import fileinput
#import Queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import string
from random import randrange, uniform
from random import *


def	Scan(IP,Port):
    msg = MIMEMultipart('related')
    min_char = 20
    max_char = 25
    X = "noreply@linux.com"
    Y = "lea.jacobs@startasys.com"
    
	  
    s = socket.socket()
    #s.settimeout(10)
    #print("[+]SCANNING FOR SMTP UNDER: "+IP+" WITH PORT: "+Port+"\n")
    
    try:
        HOSTNAME = socket.gethostbyaddr(IP)
        HOSTNAME = HOSTNAME[0] if HOSTNAME else 'UNKNOWN HOST'
    except:
        HOSTNAME = 'UNKNOWN HOST'
        
    allchar = string.ascii_letters + string.digits
    rara = "".join(choice(allchar) for x in range(randint(min_char, max_char)))
    X = rara+"@"+HOSTNAME if HOSTNAME else rara+"@"+rara+".linux.com"  
    msg['From'] = "KEYWORD"+"<%s>"%(X)
    msg['To'] = Y  
    msg['Subject'] = "FOUND [ "+IP+":"+Port+" ]"  
    inputmail = "[ "+IP+":"+Port+" ("+HOSTNAME+") ]"
    msg.attach(MIMEText(inputmail, 'plain'))

    str1 = ("[+]SCANNING FOR SMTP UNDER: "+IP+" ("+HOSTNAME+"): WITH PORT: "+Port+"\n")
    try:
        s.connect((IP,int(Port)))
        socket.setdefaulttimeout(2)
        ans = s.recv(1024)
        #print('A '+IP)
        if ("220" in ans) or ("250" in ans):
            #print('B '+IP)
            smtpserver = smtplib.SMTP(IP,int(Port), timeout=2)
            r = smtpserver.docmd("Mail From:",X)
            a = str(r)
            if ("250" in a) or ("200" in a):
                r = smtpserver.docmd("RCPT TO:",Y)
                a = str(r)
                if ("250" in a) or ("200" in a):
                    smtpserver.sendmail(msg['From'], msg['To'], msg.as_string())
                    #open('found.txt','a+').write(IP+":"+Port+" ("+HOSTNAME+") \n")
                    str2 = ("[+]SMTP FOUND ON: "+IP+" ("+HOSTNAME+"):"+Port+"\n")
                else:
                    str2 = ("[+]SMTP NOT FOUND ON: "+IP+" ("+HOSTNAME+")\n")    
            else:
                str2 = ("[+]SMTP NOT FOUND ON: "+IP+" ("+HOSTNAME+")\n")
        else:
            #print('B '+IP)
            str2 = ("[+]SMTP NOT FOUND ON: "+IP+" ("+HOSTNAME+")\n")
    except:
        str2 = ("[+]SMTP NOT FOUND ON: "+IP+" ("+HOSTNAME+")\n")
    return str1,str2


# synchronised queue
#queue = Queue.Queue(0)    # 0 means no maximum size

# do stuff to initialise queue with strings
# representing os commands
#queue.put('sleep 1')
#queue.put('echo Sleeping..')
# etc
# or use python to generate commands, e.g.

def send():
    pool = ProcessPoolExecutor(int(sys.argv[2])) ## Replace thread with ProcessPoolExecutor for later try
    results = []
    with open(sys.argv[1]) as fp:
        for host in fp:
            host = host.replace("\r","")
            host = host.replace("\n","")
            str = host
            index1 = str.find(':')
            elem1 = str[0:index1]
            elem2 = str[index1+1:]
            results.append(pool.submit(Scan,elem1, elem2))
            #print(Scan(elem1,elem2))

    for r in as_completed(results, timeout=None):
        a,b = r.result()
        print(a)
        print(b)
		            #queue.put(Scan(elem1,elem2))
	
send()

#def go():
#  while True:
#    try:
#      # False here means no blocking: raise exception if queue empty
#      command = queue.get(False)
      # Run command.  python also has subprocess module which is more
      # featureful but I am not very familiar with it.
      # os.system is easy :-)
#    except Queue.Empty:
#      return

#for i in range(1):   # change this to run more/fewer threads
#  threading.Thread(target=go).start()        	
