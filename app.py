import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
#import plotly.offline as py

from sklearn.cluster import KMeans

import matplotlib
import numpy as np
import pandas as pd
import datetime
import xlrd

import re


st.set_page_config(page_title='Australia renewables',layout='wide')

st.title('Analysis of Australian renewables market')

st.sidebar.title('Select filters')

### Allocation per investor

st.header('Allocation of PF portfolio per investor - project finance Australia')

pfaustralasia = pd.read_excel(r'Lenders_165_all_Australia.xlsx',header=1).iloc[:,:6]

sectorlist = pfaustralasia['Sector'][0].split('|')

newlist = []
for sector in sectorlist:
    sector = sector.split(':')[0].replace(' ','',1)
    newlist.append(sector)
sectorlist = newlist

def extractsector(string):
    allocation = []
    i=0
    for sector in sectorlist:
        if string.find(sector) == -1:
            allocation.append(0)
        else:
            allocation.append(re.findall('[0-9]+',string)[i])
            i=i+1
    return allocation

pfaustralasia['Allocation']=pfaustralasia['Sector'].apply(extractsector)
i=0
for sector in sectorlist:
    pfaustralasia[sectorlist[i]] = pfaustralasia['Allocation'].apply(lambda x: x[i])
    pfaustralasia[sectorlist[i]] = pd.to_numeric(pfaustralasia[sectorlist[i]])/100
    i = i+1


minsize = st.number_input('Minimum invested size filter',min_value=0,value=100)
pfaustralasiatable = pfaustralasia[pfaustralasia['Lent amount  (USD)m']>minsize]
pfaustralasiatable = pfaustralasia.drop(['Asset location','Sector','Allocation','Origin','Deal type'],axis=1)

st.subheader('Scroll to the right to see the portfolio splits')
st.write(pfaustralasiatable.style
    .background_gradient(cmap='viridis',subset=sectorlist)
    .format('{:.2%}',subset=sectorlist))

for sector in sectorlist:
    pfaustralasia[sector+' amount'] = pfaustralasia['Lent amount  (USD)m'] * pfaustralasia[sector]


st.subheader('Split of Australian pf market - lent amount since 2016')

selectsplit = st.radio('Select split by bank or by sector',['Bank','Sector'],index=1)

treemapdf = pfaustralasia[['Name','Energy amount','Environment amount','Other amount','Power amount','Renewables amount','Social Infrastructure amount','Telecommunications amount','Transport amount']]
treemapdf.set_index('Name',inplace=True)
treemapdf = treemapdf.stack()
treemapdf = pd.DataFrame(treemapdf)
treemapdf.reset_index(inplace=True)
treemapdf.columns=['Name','Sector','Lent amount  (USD)m']

if selectsplit == 'Bank':
    fig = px.treemap(treemapdf,path=['Name','Sector'],values='Lent amount  (USD)m')
else:
    fig = px.treemap(treemapdf,path=['Sector','Name'],values='Lent amount  (USD)m')

st.plotly_chart(fig)


#Data acquisition
deallist = pd.read_excel(r'360__Projects.xlsx',header=1)
bankdeal = pd.read_excel(r'Facilities_82.xlsx',header=1)


deallist = deallist[['Transaction Name','States/provinces']]


bankdeal.iloc[:,0:2] = bankdeal.iloc[:,0:2].fillna(method='pad')
bankdeal.dropna(axis=0,subset=['Name'],inplace=True)
bankdeal = bankdeal.iloc[:,:11]

bankdeal.rename(columns={'Unnamed: 0':'Bank'},inplace=True)


bankdeal = pd.merge(bankdeal,deallist,left_on='Name',right_on='Transaction Name')

deallocations = pd.read_excel(r'20200616 MLF 2020-2021.xlsx',sheet_name='Project_Locations')


bankdeal = pd.merge(bankdeal,deallocations,left_on='Name',right_on='Name')
bankdealunfiltered = bankdeal

### Percentage solar


bankdealunfiltered['Solar'] = bankdealunfiltered['Lent amount (USD)m'][bankdealunfiltered['Sub-sector']=='Solar PV']

pctsolar = bankdealunfiltered.groupby(by='Bank').sum()
pctsolar['Solar %'] = pctsolar['Solar'] / pctsolar['Lent amount (USD)m']
pctsolar.reset_index(inplace=True)


## Selection filter widgets in the sidebar

#selectyear = st.sidebar.multiselect('Select years',(2016,2017,2018,2019,2020),default=[2016,2017,2018,2019,2020])
selectyear = st.sidebar.slider('Select period',min_value=2016,max_value=2020,value=(2016,2020),step=1)
selectstate = st.sidebar.multiselect('Select states',list(bankdeal['States/provinces'].unique()),default=list(bankdeal['States/provinces'].unique()))

bankdeal = bankdeal[bankdeal['Total lent amount (USD)m']>0]
bankdeal['Year'] = pd.DatetimeIndex(bankdeal['Financial close date']).year
bankdeal = bankdeal[bankdeal['Year'].isin(range(selectyear[0],selectyear[1]+1))]
bankdeal = bankdeal[bankdeal['States/provinces'].isin(selectstate)]

bankdeal = bankdeal[['Bank','Origin','Name','Sub-sector','Deal type','Lent amount (USD)m','Total lent amount (USD)m','States/provinces','Year','lat','lon']]

### Data display

st.header('Deals by bank')
mydeal = st.text_input(r'Looking for a particular deal or a particular bank? Search in the textbox below.')
mysector = st.multiselect('Select subsectors',list(bankdeal['Sub-sector'].unique()),default=['Solar PV','Onshore wind'])
#bankdeal['Bubble Size']=1
if mydeal == '':
    st.write(bankdeal)
    fig = px.scatter_mapbox(bankdeal[bankdeal['Sub-sector'].isin(mysector)],lat='lat',lon='lon',hover_name='Name', color = 'Sub-sector',hover_data=['Name','Sub-sector','Deal type'],zoom=3.2,center=dict(lat=-30,lon=133),size='Total lent amount (USD)m',size_max=15,height=600)
    fig.update_layout(mapbox_style='open-street-map',title='Renewable projects in Australia - bubble size is total debt amount')
    st.plotly_chart(fig,use_container_width=True)
else:
    st.write(bankdeal[(bankdeal['Name'].str.contains(mydeal)) | (bankdeal['Bank'].str.contains(mydeal)) & (bankdeal['Sub-sector'].isin(mysector))])
    fig = px.scatter_mapbox(bankdeal[(bankdeal['Name'].str.contains(mydeal)) | (bankdeal['Bank'].str.contains(mydeal)) & (bankdeal['Sub-sector'].isin(mysector))],lat='lat',lon='lon',hover_name='Name', color = 'Sub-sector',hover_data=['Name','Sub-sector','Deal type'],zoom=3.2,center=dict(lat=-30,lon=133),size='Total lent amount (USD)m',size_max=10,height=600)
    fig.update_layout(mapbox_style='open-street-map',title='Renewable projects in Australia - bubble size is total debt amount')
    st.plotly_chart(fig,use_container_width=True)

def getbanklist(bank):
    listdeals = bankdeal['Name'][bankdeal['Bank'].str.contains(bank)]
    return bankdeal['Bank'][bankdeal['Name'].isin(listdeals)].value_counts()

if mydeal != '' and getbanklist(mydeal).empty==False:
    pfaustralasia = getbanklist(mydeal)[1:]
    st.subheader('Investors most seen on {} deals.'.format(mydeal))
    fig = px.bar(pfaustralasia,x=pfaustralasia.index,y='Bank',color='Bank')
    fig.update_layout(title='Number of deals per investor',coloraxis_showscale=False,template='ggplot2',xaxis_title=None,yaxis_title='# deals')
    st.plotly_chart(fig,use_container_width=True)


## MLF analysis

st.header('Marginal loss factors 2020-2021')

myproject = st.text_input(r'Looking for a particular project? Search in the textbox below.')

mlfpfaustralasia = pd.read_excel(r'20200616 MLF 2020-2021.xlsx',sheet_name='MLF_table')

mlffilter = st.slider('MLF percentage',70,115,(70,115))
mlfpfaustralasia = mlfpfaustralasia[(mlfpfaustralasia['2020/21 MLF']>mlffilter[0]/100)&(mlfpfaustralasia['2020/21 MLF']<mlffilter[1]/100)&(mlfpfaustralasia['Generator'].str.contains(myproject))]
mlfpfaustralasia = mlfpfaustralasia[['State','Generator','Sub-asset class','2020/21 MLF','2019/20 MLF','Regional rank','Regional % bottom rank','lat','lon','Transaction Name']]

showdata = st.checkbox('Show data')
if showdata:
    st.write(mlfpfaustralasia)

sectorpick = st.multiselect('Select sub-asset class',['Solar','Wind','Other'],default=['Solar','Wind'])
mlfpfaustralasiafilter = mlfpfaustralasia[mlfpfaustralasia['Sub-asset class'].isin(sectorpick)]

mlfpfaustralasiafilter['Bubble Size']=1

fig = px.scatter_mapbox(mlfpfaustralasiafilter,lat='lat',lon='lon',hover_name='Generator', color = '2020/21 MLF',color_continuous_scale='Inferno',hover_data=['2020/21 MLF','2019/20 MLF','Sub-asset class'],zoom=3.2,center=dict(lat=-30,lon=133),size='Bubble Size',size_max=10,height=600)
fig.update_layout(mapbox_style='open-street-map',title='MLF in Australia')
st.plotly_chart(fig,use_container_width=True)


## MLF Analysis - distribution of MLF per deal type


fig = px.box(mlfpfaustralasia,x='Sub-asset class',y='2020/21 MLF',hover_data=['Generator'],color='Sub-asset class',points='all')
fig.update_layout(title='MLF distribution by Sub-asset class',template='ggplot2')
st.plotly_chart(fig,use_container_width=True)


showstats = st.checkbox('Show stats')
if showstats:
    st.subheader('MLF distribution by Sub-asset class - Data')
    statedetail = st.checkbox('Show detail by State')
    if statedetail:
        st.write(mlfpfaustralasia[['2020/21 MLF','Sub-asset class','State']].groupby(['Sub-asset class','State']).describe())
    else:
        st.write(mlfpfaustralasia[['2020/21 MLF','Sub-asset class','State']].groupby(['Sub-asset class']).describe())


##MLF per bank

st.subheader('List of banks with lowest MLF')
st.markdown('Using (1-MLF)* Amount Lent / Total Amount Lent in renewables on most recent projects with MLF')

newpfaustralasia = pd.merge(bankdealunfiltered,mlfpfaustralasia,left_on='Name',right_on='Transaction Name')

newpfaustralasia['MLF amount'] = (1-newpfaustralasia['2020/21 MLF']) * newpfaustralasia['Lent amount (USD)m']

mlfranking = newpfaustralasia[['Bank','Lent amount (USD)m','MLF amount']].groupby(by='Bank').sum()
mlfranking['Average MLF'] = 1- mlfranking['MLF amount'] / mlfranking['Lent amount (USD)m']
mlfranking.reset_index(inplace=True)

st.write(mlfranking.sort_values(by='Average MLF',ascending=True).style.background_gradient(cmap='RdYlGn',subset='Average MLF'))

## Electricity prices

st.header('Electricity prices')

pricespfaustralasia = pd.read_csv(r'AER_Spot prices_Weekly VWA spot prices regions DATA_3_20200706115427.CSV')
#pricespfaustralasia.set_index('Week commencing',inplace=True)
pricespfaustralasia.columns = ['Week commencing','Queensland','New South Wales','Victoria','South Australia','Tasmania']

showprices = st.checkbox('Show data',key=1)
if showprices:
    st.write(pricespfaustralasia)

pricespfaustralasia = pricespfaustralasia[::-1]

pricespfaustralasia['Week']=pd.to_datetime(pricespfaustralasia['Week commencing'],errors='ignore',dayfirst=True)
pricespfaustralasia['Year']=pd.DatetimeIndex(pricespfaustralasia['Week']).year

yearrange = st.slider('Select your date range',min_value=int(pricespfaustralasia['Year'].min()),max_value=int(pricespfaustralasia['Year'].max()),value=(2008,2020),step=1)

st.markdown('The colorscale below will range from Breakeven price + 1 Buffer to Breakeven price + 2 buffers')
breakprice = st.number_input('Breakeven price',value=35,step=5)
buffer = st.number_input('Buffer over breakeven price',value=10,step=5)


pricespfaustralasia=pricespfaustralasia[(pricespfaustralasia['Year']>=yearrange[0])&(pricespfaustralasia['Year']<=yearrange[1])]

fig = go.Figure(data=go.Heatmap(z=pricespfaustralasia[['Queensland','New South Wales','Victoria','South Australia','Tasmania']].transpose(),x=pricespfaustralasia['Week'],y=['Queensland','New South Wales','Victoria','South Australia','Tasmania'],colorscale=['Red','Green'],zmax=breakprice+2*buffer,zmin=breakprice+buffer))
#fig.update_xaxes(autorange='reversed')
fig.update_layout(title='Price evolution')
st.plotly_chart(fig,use_container_width=True)


# Distribution of prices

pricespfaustralasia=pricespfaustralasia[['Queensland','New South Wales','Victoria','South Australia','Tasmania']]

st.subheader('Distribution of prices')

showpricestats = st.checkbox('Show data',key=2)
if showpricestats:
    st.write(pricespfaustralasia.describe())



## Deal volume per year

bankdealcount = pd.DataFrame(bankdeal.groupby(by=['Year','Sub-sector'])['Name'].count())
bankdealcount.reset_index(inplace=True)


st.header('Deal volume per year')
volcount = st.radio('Select indicator',['Deal volume','Deal count'])

if volcount == 'Deal volume':
    fig = px.bar(bankdeal,x='Year',y='Lent amount (USD)m',color='Sub-sector',hover_name='Name',template='plotly_white')
    fig.update_layout(barmode='stack',xaxis=dict(tickmode='array',tickvals=[2016,2017,2018,2019,2020]))
    st.plotly_chart(fig,use_container_width=True,xaxis=dict(showlabel=False))
else:
    fig = px.bar(bankdealcount,x='Year',y='Name',color='Sub-sector',hover_name='Name',template='plotly_white')
    fig.update_layout(barmode='stack',xaxis=dict(tickmode='array',tickvals=[2016,2017,2018,2019,2020]),yaxis=dict(title='Deal count'))
    st.plotly_chart(fig,use_container_width=True,xaxis=dict(showlabel=False))


## Comparison of deals between banks

st.header('Compare deals across banks')

selectsector = st.multiselect('Select sub-sectors',list(bankdeal['Sub-sector'].unique()),default=['Onshore wind','Solar PV','Portfolio'])

bank1 = st.selectbox('Bank 1',bankdeal['Bank'].sort_values().unique(),index=0)
bank2 = st.selectbox('Bank 2',bankdeal['Bank'].sort_values().unique(),index=1)
bank3 = st.selectbox('Bank 3',bankdeal['Bank'].sort_values().unique(),index=2)

banklist = [bank1,bank2,bank3]

bankdealfilter = bankdeal[(bankdeal['Bank'].isin(banklist))&(bankdeal['Sub-sector'].isin(selectsector))]

# trace1 = bankdealfilter[(bankdealfilter['Bank']==bank1) & (bankdealfilter['Sub-sector'].isin(selectsector))]
# trace2 = bankdealfilter[(bankdealfilter['Bank']==bank2) & (bankdealfilter['Sub-sector'].isin(selectsector))]
#
# fig = go.Figure(data=[
#     go.Bar(name=bank1,x=trace1['Year'],y=trace1['Lent amount (USD)m'],text=trace1['Name']),
#     go.Bar(name=bank2,x=trace2['Year'],y=trace2['Lent amount (USD)m'],text=trace2['Name'])
#         ])
#
# fig.update_layout(barmode='group',template='plotly_white',xaxis=dict(tickmode='array',tickvals=[2016,2017,2018,2019,2020]))

fig = px.scatter_mapbox(bankdealfilter,lat='lat',lon='lon',hover_name='Name', color = 'Bank',hover_data=['Name','Sub-sector','Deal type'],zoom=3.2,center=dict(lat=-30,lon=133),size='Lent amount (USD)m',size_max=15,height=600)
fig.update_layout(mapbox_style='open-street-map',title='Bank comparison - Bubble size is amount lent (USD)m',legend_orientation='h')
st.plotly_chart(fig,use_container_width=True)


st.header('Deal volume by investor - renewables')
fig = px.bar(bankdeal,x='Bank',y='Lent amount (USD)m',color='Sub-sector',hover_name='Name',template='plotly_white')
fig.update_layout(barmode='stack',xaxis={'categoryorder':'total descending'},xaxis_title=None,)
st.plotly_chart(fig,use_container_width=True,xaxis=dict(showlabel=False))
#py.plot(fig,filename='bar.html')



st.header('Deal by deal analysis')
dimensionoptions = ['Country > Bank > Sub Asset Class > Deal detail','Sub Asset Class > Deal detail > Bank','Sub Asset Class > Bank > Deal detail','Sub Asset Class > Deal type > Bank > Deal detail','State > Sub Asset Class > Bank > Deal detail']
selectdimensions = st.selectbox('Select dimensions, the sunburst chart below will adjust.',dimensionoptions,index=0)

if selectdimensions == dimensionoptions[0]:
    st.subheader('Deal split per bank')
    st.text(selectdimensions)
    fig = px.sunburst(bankdeal,path=['Origin','Bank','Sub-sector','Name'],values='Lent amount (USD)m')
elif selectdimensions == dimensionoptions[1]:
    st.header('Split of banks by deal type - club by deal')
    st.text(selectdimensions)
    fig = px.sunburst(bankdeal,path=['Sub-sector','Name','Bank'],values='Lent amount (USD)m')
elif selectdimensions == dimensionoptions[2]:
    st.header('Split of banks by deal type - deals by bank')
    st.text(selectdimensions)
    fig = px.sunburst(bankdeal,path=['Sub-sector','Bank','Name',],values='Lent amount (USD)m')
elif selectdimensions == dimensionoptions[3]:
    st.header('Split of deals by type')
    st.text(selectdimensions)
    fig = px.sunburst(bankdeal,path=['Sub-sector','Deal type','Bank','Name',],values='Lent amount (USD)m')
elif selectdimensions == dimensionoptions[4]:
    st.header('Split of deals by state')
    st.text(selectdimensions)
    fig = px.sunburst(bankdeal,path=['States/provinces','Sub-sector','Bank','Name',],values='Lent amount (USD)m')

st.plotly_chart(fig,use_container_width=True,xaxis=dict(showlabel=False))

st.header('Distribution of deals')
distribchoice = st.radio('Select distribution axis.',['Total debt size by deal','Ticket size by deal'],index=0)


distribfilter = st.multiselect('Select sub-sectors',list(bankdeal['Sub-sector'].unique()),default=['Onshore wind','Solar PV'])

bankdealsfilter = bankdeal[bankdeal['Sub-sector'].isin(distribfilter)]

if distribchoice == 'Total debt size by deal':
    bankdealsfilter=bankdealsfilter.groupby(['Name','Sub-sector']).mean()
    bankdealsfilter.reset_index(inplace=True)
    fig = px.box(bankdealsfilter,x='Sub-sector',y='Total lent amount (USD)m',hover_data=['Name'],color='Sub-sector',points='all')
else:
    fig = px.box(bankdealsfilter,x='Sub-sector',y='Lent amount (USD)m',hover_data=['Name','Bank'],color='Sub-sector',points='all')

fig.update_layout(title='Distribution of deal and ticket size',template='ggplot2')
st.plotly_chart(fig,use_container_width=True)



### Consolidated dataframe

pfaustralasia = pfaustralasia[pfaustralasia['Lent amount  (USD)m']>minsize]

bigdf = pd.merge(mlfranking,pfaustralasia,left_on='Bank',right_on='Name')

bigdf = pd.merge(bigdf,pctsolar,left_on='Bank',right_on='Bank')
bigdf = bigdf[['Bank','Solar %','Average MLF','Renewables']]


bigdf.set_index('Bank',inplace=True)
st.title('K-means clustering analysis')
st.subheader('Tip: you can rotate the chart and zoom')

clustnum = st.number_input('Number of clusters',min_value=2,max_value=6,value=3)

kmeans = KMeans(n_clusters=clustnum,random_state=0).fit(bigdf)


bigdf.reset_index(inplace=True)

shownames = st.checkbox('Display bank names on chart')
if shownames:
    textnames = 'Bank'
else:
    textnames=None


fig = px.scatter_3d(bigdf,x='Solar %',y='Average MLF',z='Renewables',opacity=0.6,hover_name='Bank',hover_data=['Solar %','Average MLF','Renewables'],text=textnames,color=kmeans.labels_,color_continuous_scale='Portland')
fig.update_layout(template='ggplot2',title='Bank segmentation in Australia renewables market',coloraxis_showscale=False)

st.plotly_chart(fig,use_container_width=True)
