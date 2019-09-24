import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import folium
from folium.features import DivIcon
import matplotlib

df = pd.read_csv('mural_data.csv',names=['name','lat','lng','size'])
#df.head()

folium_map = folium.Map(location=[(df.lat.min()+df.lat.max())/2, (df.lng.min()+df.lng.max())/2],
                        zoom_start=15,
                        tiles="CartoDB dark_matter"
                        )

for i in range(0, len(df)):
    folium.CircleMarker(location=(df.loc[i]['lat'], df.loc[i]['lng']),
                            radius=float(df.loc[i]['size']),
                            color='white',
                            popup=df.loc[i]['name'] + ':' + str(df.loc[i]['size']),
                            tooltip=df.loc[i]['name'],
                            fill=True).add_to(folium_map)
                            
                            
folium_map.save('temp_white.html')


df2 = df.sort_values(by='size').reset_index()
#df2.head()                        

#plt.style.use('dark_background')
plt.rcParams['savefig.facecolor'] = '#0f0f0f'
fig, ax = plt.subplots(figsize=(20,8),facecolor='#0f0f0f')

df2.plot(x='name', y='size', kind='bar', color='white', alpha=0.2, linewidth=5 ,edgecolor='white', ax=ax,
        width=0.60);
ax.set_facecolor('#0f0f0f')
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.tick_params(axis='both', which='major', labelsize=16)
ax.tick_params(axis='both', which='minor', labelsize=16)

for i in range(len(df2)):
    plt.text(i-0.35, df2.loc[i]['size']+0.5, str(df2.loc[i]['size']), color='white', fontsize=12)
    
plt.tight_layout()
fig.savefig('myimage.svg',format='svg',dpi=2000 ,facecolor='#0f0f0f')
fig.savefig('myimage.png',dpi=1000 ,facecolor='#0f0f0f')
