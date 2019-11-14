import pandas as pd
import numpy as np
import json
import re

regex = re.compile(r'\s+')

all_vendor_data = []
with open('LLDP_Discovery.txt') as f:
    vendor,hostname,ip_addr = None, None, None
    vendor_data = []
    for line in f:
        clean_line = line.strip()
        if clean_line:
            if 'command completed' in clean_line.lower():
                vendor_table = []
                for row in vendor_data:
                    vendor_table.append({'localport' : row[0], 'neighborport':row[1], 'systemname':row[2]})
                all_vendor_data.append({
                    'Vendor': vendor,
                    'Hostname': hostname,
                    'IPAddress':ip_addr,
                    'lldp': vendor_table
                })
                vendor,hostname,ip_addr,vendor_data = None, None, None, []
                continue
            if not vendor:
                vendor = 'AAA' if 'AAA' in clean_line else 'BBB' if 'BBB' in clean_line else 'CCC' if 'CCC' in clean_line else None
                continue
            elif 'Link Layer Discovery Protocol' in clean_line or 'nearest-bridge' in clean_line:
                continue
            if vendor == 'AAA':
                if 'hostname' in clean_line.lower():
                    hostname = clean_line.split(':')[1].strip()
                elif 'ip address' in clean_line.lower():
                    ip_addr = clean_line.split(':')[1].strip()
                split_results = regex.split(clean_line)
                if len(split_results) == 4:
                    vendor_data.append([split_results[0], split_results[2], split_results[3]])
            elif vendor == 'BBB':
                if 'hostname' in clean_line.lower():
                    hostname = clean_line.split(':')[1].strip()
                elif 'ip address' in clean_line.lower():
                    ip_addr = clean_line.split(':')[1].strip()
                split_results = regex.split(clean_line)
                if len(split_results) == 4:
                    vendor_data.append([split_results[0], split_results[2], split_results[1]])
            elif vendor == 'CCC':
                if 'hostname' in clean_line.lower():
                    hostname = clean_line.split(':')[1].strip()
                elif 'ip address' in clean_line.lower():
                    ip_addr = clean_line.split(':')[1].strip()
                split_results = regex.split(clean_line)
                if len(split_results) >=6 and len(split_results) <= 10:
                    port = ' '.join(split_results[4:-1])
                    vendor_data.append([split_results[0], port.split(',')[0].strip(),split_results[-1]])
                    
                    
                    
                    
final_output = {'scandevice': [{'scanlldp': all_vendor_data}]}
with open('out.json', 'w') as f:
    json.dump(final_output, f,indent=2)
