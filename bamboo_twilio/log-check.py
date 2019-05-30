import json
import requests
from requests.auth import HTTPBasicAuth
from twilio.rest import Client
import time
import sys
import os
import datetime
#URL: http://34.229.165.221:8085/
#Username: testuser
#password: testpassword

account_sid = 'AC44ad7addd81382eb21f19cecfe8c5634'
auth_token = '1ef1718fcc720eeea5bacdb7a9a73677'

from_number = '+13236901080'
to_numbers = ['+13236903633','+14016543129']

def call(client, message, from_number, to_numbers):
    ## Below code writes xml to a location which will be accessible on iternet as URL. Then twilio will call it to send call.
    #with open('temp.xml', 'w') as f:
    #    f.write('<?xml version="1.0" encoding="UTF-8"?>\n \
    #                <Response>\n \
    #                <Say voice="alice">%s</Say>\n \
    #                <Play>http://demo.twilio.com/docs/classic.mp3</Play>\n \
    #            </Response>'%message)
                
    for number in to_numbers:
        call = client.calls.create(
                    url='http://demo.twilio.com/docs/voice.xml',
                    to=number,
                    from_=from_number
                )

        logger.info('Called number : ',number,' SID : ',call.sid)
        
    ## Removing custom xml with custom message after call is completed.
    #os.remove('temp.xml')

def send_message(client, body_message, from_number, to_numbers):
    for number in to_numbers:
        message = client.messages \
        .create(
             body=body_message,
             from_=from_number,
             to=number
         )

        print('Message sent to : ',number,' SID : ',message.sid)

if __name__ == '__main__':
    log_location = sys.argv[1]
    client = Client(account_sid, auth_token)        
    latest_file = os.listdir(log_location)[0]
    latest_time = datetime.datetime.fromtimestamp(os.path.getmtime(latest_file))
    for other_file in os.listdir(log_location)[1:]:
        other_time = datetime.datetime.fromtimestamp(os.path.getmtime(other_file))
        if other_time  > latest_time:
            latest_time = other_time
            latest_file = other_file
            
    error_check_flag = False
    err_count = 0
    error_line_locations = []
    with open(os.path.join(log_location, latest_file), 'r') as f:
        for i, line in enumerate(f):
            if 'an error has occured' in line or 'process terminated' in line:
                error_check_flag = True
                err_count += 1
                error_line_locations.append(i+1)
        if error_check_flag: 
                print(latest_file, err_count, error_line_locations)       
                body_message = '%s has failures. Error Count : %d. Error Line Locations : %s'\
                    %(os.path.join(log_location, latest_file),err_count, str(error_line_locations))
                send_message(client, body_message, from_number, to_numbers)
                call(client, body_message, from_number, to_numbers)
