import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('nuclear.csv', delimiter=',')

countries_shortNames = [['UNITED STATES OF AMERICA', 'USA'], \
                        ['RUSSIAN FEDERATION', 'RUSSIA'], \
                        ['IRAN, ISLAMIC REPUBLIC OF', 'IRAN'], \
                        ['KOREA, REPUBLIC OF', 'SOUTH KOREA'], \
                        ['TAIWAN, CHINA', 'CHINA']]
for shortName in countries_shortNames:
    df = df.replace(shortName[0], shortName[1])

import folium
import matplotlib.cm as cm
import matplotlib.colors as colors

latitude, longitude = 40, 10.0
map_world_NPP = folium.Map(location=[latitude, longitude], zoom_start=2)

viridis = cm.get_cmap('viridis', df['NumReactor'].max())
colors_array = viridis(np.arange(df['NumReactor'].min() - 1, df['NumReactor'].max()))
rainbow = [colors.rgb2hex(i) for i in colors_array]

for nReactor, lat, lng, borough, neighborhood in zip(df['NumReactor'].astype(int), df['Latitude'].astype(float),
                                                     df['Longitude'].astype(float), df['Plant'], df['NumReactor']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=3,
        popup=label,
        color=rainbow[nReactor - 1],
        fill=True,
        fill_color=rainbow[nReactor - 1],
        fill_opacity=0.5).add_to(map_world_NPP)

# 在地图上显示
map_world_NPP.save('world_map.html')  # 保存为 HTML 文件
# 然后打开world_map.html 文件 可以看到


countries = df['Country'].unique()
df_count_reactor = [[i, df[df['Country'] == i]['NumReactor'].sum(), df[df['Country'] == i]['Region'].iloc[0]] for i in
                    countries]
df_count_reactor = pd.DataFrame(df_count_reactor, columns=['Country', 'NumReactor', 'Region'])
df_count_reactor = df_count_reactor.set_index('Country').sort_values(by='NumReactor', ascending=False)[:20]
ax = df_count_reactor.plot(kind='bar', stacked=True, figsize=(10, 3),
                           title='The 20 Countries With The Most Nuclear Reactors in 2010')
ax.set_ylim((0, 150))
for p in ax.patches:
    ax.annotate(str(p.get_height()), xy=(p.get_x(), p.get_height() + 2))
df_count_reactor['Country'] = df_count_reactor.index
sns.set(rc={'figure.figsize': (11.7, 8.27)})
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
ax = sns.barplot(x="NumReactor", y="Country", hue="Region", data=df_count_reactor, dodge=False, orient='h')
ax.set_title('2010年拥有最多核反应堆的20个国家', fontsize=16)
ax.set_xlabel('Reactors', fontsize=16)
ax.set_ylabel('')
ax.legend(fontsize='14')

plt.show()

def getMostExposedNPP(Exposedradius):
    df_pop_sort = df.sort_values(by=str('p10_' + str(Exposedradius)), ascending=False)[:10]
    df_pop_sort['Country'] = df_pop_sort['Plant'] + ',\n' + df_pop_sort['Country']
    df_pop_sort = df_pop_sort.set_index('Country')
    df_pop_sort = df_pop_sort.rename(
        columns={str('p90_' + str(Exposedradius)): '1990', str('p00_' + str(Exposedradius)): '2000',
                 str('p10_' + str(Exposedradius)): '2010'})
    df_pop_sort = df_pop_sort[['1990', '2000', '2010']] / 1E6
    ax = df_pop_sort.plot(kind='bar', stacked=False, figsize=(10, 4))
    ax.set_ylabel('Population Exposure in millions', size=14)
    ax.set_title(
        'Location of nuclear power plants \n with the most exposed population \n within ' + Exposedradius + ' km radius',
        size=16)
    print(df_pop_sort['2010'])

getMostExposedNPP('30')


latitude, longitude = 40, 10.0
map_world_NPP = folium.Figure(width=100, height=100)
map_world_NPP = folium.Map(location=[latitude, longitude], zoom_start=2)

for nReactor, lat, lng, borough, neighborhood in zip(df['NumReactor'].astype(int), df['Latitude'].astype(float),
                                                     df['Longitude'].astype(float), df['Plant'], df['NumReactor']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.Circle(
        [lat, lng],
        radius=30000,
        popup=label,
        color='grey',
        fill=True,
        fill_color='grey',
        fill_opacity=0.5).add_to(map_world_NPP)

Exposedradius = '30'
df_sort = df.sort_values(by=str('p10_' + str(Exposedradius)), ascending=False)[:10]

for nReactor, lat, lng, borough, neighborhood in zip(df_sort['NumReactor'].astype(int),
                                                     df_sort['Latitude'].astype(float),
                                                     df_sort['Longitude'].astype(float), df_sort['Plant'],
                                                     df_sort['NumReactor']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.25).add_to(map_world_NPP)

for nReactor, lat, lng, borough, neighborhood in zip(df_sort['NumReactor'].astype(int),
                                                     df_sort['Latitude'].astype(float),
                                                     df_sort['Longitude'].astype(float), df_sort['Plant'],
                                                     df_sort['NumReactor']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.Circle(
        [lat, lng],
        radius=30000,
        popup=label,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.25).add_to(map_world_NPP)

# 在地图上显示
map_world_NPP.save('world_map2.html')  # 保存为 HTML 文件
