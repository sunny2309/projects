import json
import requests
from requests.auth import HTTPBasicAuth
from twilio.rest import Client
import time
import sys

#URL: http://34.229.165.221:8085/
#Username: testuser
#password: testpassword

account_sid = 'AC44ad7addd81382eb21f19cecfe8c5634'
auth_token = '1ef1718fcc720eeea5bacdb7a9a73677'
plans_to_check = sys.argv[1].split(',')

to_numbers = ['+13236903633','+14016543129']

res = requests.get('http://34.229.165.221:8085/rest/api/latest/result.json?os_authType=basic', 
                   auth=HTTPBasicAuth('testuser', 'testpassword'))
result = json.loads(res.content.decode())

for plan in result['results']['result']:
    if (plan['state'] == 'Failed' or plan['buildState'] == 'Failed') and plan['plan']['shortName'] in plans_to_check:
        print('Plan %s has failed. Sending Message'%plan['plan']['shortName'])
        
        client = Client(account_sid, auth_token)
        for number in to_numbers:
            message = client.messages \
            .create(
                 body='%s has failed'%plan['plan']['shortName'],
                 from_='+13236901080',
                 to=number
             )

            print('Message sent to : ',number,' SID : ',message.sid)
