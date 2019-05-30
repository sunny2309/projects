import requests
import sys
import json
import pandas as pd

rooms_condo_landed = {'Bedrooms': 0, 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9'}
rooms_hbd = {'Bedrooms': '', '1-Room':'1-Room', '2-Room':'1-Room', '3-Room':'3-Room', '4-Room':'4-Room',
             '5-Room':'5-Room', 'Executive':'Executive','Jumbo':'Jumbo'}
transtype = {'Rent': 'Rent', 'Sale':'Sale'}
#type = {'Rent': 'Rent', 'Sale':'Sale'}
category_dict = {'HDB': 'h', 'Condominium':'r', 'Landed':'l','Commercial':'c','Industrial':'i'}
prop_type = {'Select One':'', 'Office':'Office', 'Shop':'Shop', 'Shop House':'Shop House','Medical':'Medical',
            'Detached House':'Detached House','Semi-Detached House':'Semi-Detached House',
            'Terrace House':'Terrace House','Factory':'Factory (All Types)', 'Warehouse':'Warehouse'}
            
with open(sys.argv[1]) as f:  
    header = ['Data Available','Sale/Rent','property_type','project_name','address','floor_area', 'area_unit', 'value','dollar_unit','value_psf', 'price_unit', 'Asset ID','Property Asset ID URL','Property Result URL']  ## Column Names
    final_result = []
    for i, line in enumerate(f):
        if i == 0:
            continue
        l = map(lambda x : x.strip(),line.split(','))
        l = list(l)
        if len(l) == 8:
            category,address,block,floor,unit,area,room,sale_rent = l
            if '(' in area:
                modified_area = area.split('(')[0]
            else:
                modified_area = area
            url1 = 'https://www.edgeprop.sg/index.php?option=com_analytica&task=search&p=%s&mode=0&type=%s&schematic=1'%(address.upper(), category_dict.get(category,''))
            res1 = requests.get(url1)
            address_content = json.loads(res1.content)
            if 'results' in address_content and len(address_content['results']) > 0 and 'id' in address_content['results'][0]:
                asset_id = address_content['results'][0]['id']
                url2 = 'https://www.edgeprop.sg/index.php?option=com_analytica&task=edgevalue&type=%s&propertytype=&block=&storey=&stack=&area2=&area=%s&rooms=%s&transtype=%s&assetid=%s'%\
                                    (sale_rent, modified_area, room, sale_rent, asset_id)
                res2 = requests.get(url2) 
                property_result = json.loads(res2.content)
                #print(property_result)
                if 'error' in property_result:
                    final_result.append(['Not Found : '+property_result['error'], sale_rent, category, '', address, area,'','','','','','','',''])
                else:
                    final_result.append(['Found', sale_rent, property_result['results']['property_type'], property_result['results']['project_name'],property_result['results']['address'],
                                         property_result['results']['floor_area'],property_result['results']['area_unit'],
                         property_result['results']['value'],property_result['results']['dollar_unit'], property_result['results']['value_psf'], property_result['results']['price_unit'], asset_id,url1,url2])
            else:
                final_result.append(['Not Found : '+address_content['error'], sale_rent, category, '', address, area,'','','','','','','',''])

final_df = pd.DataFrame(final_result, columns=header)
final_df.to_csv('result2.csv')
