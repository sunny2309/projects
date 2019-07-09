import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import unicodedata

attributes=['Crossing','Finishing','Heading Accuracy',
 'Short Passing','Volleys','Dribbling','Curve',
 'FK Accuracy','Long Passing','Ball Control','Acceleration',
 'Sprint Speed','Agility','Reactions','Balance',
 'Shot Power','Jumping','Stamina','Strength',
 'Long Shots','Aggression','Interceptions','Positioning',
 'Vision','Penalties','Composure','Marking',
 'Standing Tackle','Sliding Tackle','GK Diving',
 'GK Handling','GK Kicking','GK Positioning','GK Reflexes']

links=[]   #get all Argentinian players

for offset in ['0','60','120','180','240','300','360','420','480','540','600']:
    page=requests.get('http://sofifa.com/players?na=52&offset='+offset)
    soup=BeautifulSoup(page.content,'html.parser')
    for link in soup.find_all('a'):
        links.append(link.get('href'))
links=['http://sofifa.com'+l for l in links if 'player/'in l]
print(len(links))
#print(links)
#pattern regular expression
pattern=r"""\s*([\w\s\(\)]*?)\s*FIFA"""   #file starts with empty spaces... players name...FIFA...other stuff
for attr in attributes:
    pattern+=r""".*?(\d*\s*"""+attr+r""")"""  #for each attribute we have other stuff..number..attribute..other stuff
pat=re.compile(pattern, re.DOTALL)    #parsing multiline text

rows=[]
#links=links[10:]
for j,link in enumerate(links):
    print(j,link)
    row=[link]
    playerpage=requests.get(link)
    playersoup=BeautifulSoup(playerpage.content,'html.parser')
    text=playersoup.get_text()
    text=unicodedata.normalize('NFKD', text).encode('ascii','ignore')
    #print(text)
    a=pat.match(text.decode())
    #print(a.group(1))
    row.append(a.group(1))
    for i in range(2,len(attributes)+2):
        row.append(int(a.group(i).split()[0]))
    rows.append(row)
    #print(row[1])
df=pd.DataFrame(rows,columns=['link','name']+attributes)
df.to_csv('ArgentinaPlayers.csv',index=False)
