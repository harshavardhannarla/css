import plotly.graph_objects as go
import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import plotly.express as px
import yahooquery as yq

# add title as "Stock Tips Analysis"
st.set_page_config(layout="wide")
st.title("Telegram Stock Tips Analysis")

df = pd.DataFrame()
pd.options.plotting.backend = 'plotly'

# add a switch button
st.sidebar.markdown('## Select Dataset')
switch = st.sidebar.radio("", ["Default dataset", "Custom dataset"])

uploaded_file = None
if switch == "Custom dataset":
    uploaded_file = st.file_uploader("Upload csv file of tips to analyze your own data", type="csv")
    # convert uploaded_file to pandas dataframe
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("data_finaled.csv")

if uploaded_file is None and switch == "Custom dataset":
    st.stop()


def draw_pie_chart():

    profits = df['profit'].to_list()
    y = [sum(i > 0 for i in profits), sum(i < 0 for i in profits)]
    x = ["profit", "loss"]
    # draw pie chart using plotly using x as labels and y as values
    fig = px.pie(df, values=y, names=x, title='Number of tips that resulted in Profit/Loss Distribution')
    st.plotly_chart(fig)


def draw_no_tips_chart(timeframe_type, tip_type):

    df['date'] = pd.to_datetime(df['date'])

    if tip_type == 'Buy':
        dat = df[df['bs'] == 'BUY']
    elif tip_type == 'Sell':
        dat = df[df['bs'] == 'SELL']
    else:
        dat = df

    if timeframe_type == 'daily':
        dat = dat.groupby('date').sum().reset_index()
        fig = px.line(dat, x='date', y='count1')
        fig.update_layout(
            title_text='Daily',
            xaxis_title_text='Date',
            yaxis_title_text='Number of tips',
        )
        st.plotly_chart(fig)
    elif timeframe_type == 'monthly':
        fig = px.histogram(dat, x="date", y="count1", histfunc="sum")

        fig.update_traces(xbins_size="M1")
        fig.update_xaxes(showgrid=True, ticklabelmode="period", dtick="M1", tickformat="%b\n%Y")
        fig.update_layout(bargap=0.1,
                          title_text='Monthly',  # title of plot
                          xaxis_title_text='Month',  # xaxis label
                          yaxis_title_text='Number oF tips',  # yaxis label)
                          )
        st.plotly_chart(fig)
    elif timeframe_type == 'weekly':
        fig = go.Figure(go.Histogram(x=dat['date'], y=dat['count1'], histfunc="sum", autobinx=False,
                                     xbins=dict(size='604800000.0'
                                                )))
        fig.update_layout(
            title_text='Weekly',  # title of plot
            xaxis_title_text='Month',  # xaxis label
            yaxis_title_text='Number oF tips',)

        st.plotly_chart(fig)


stock_codes = {}
stock_codes['NIFTY50'] = '^NSEI'
stock_codes['SENSEX'] = '^BSESN'
stock_codes['RELIANCE'] = 'RELIANCE.NS'
stock_codes['ADANI ENTERPRISES'] = 'ADANIENT.NS'


def compare_profit_index(index_type):

    df['date'] = pd.to_datetime(df['date'])
    dat = df
    dat = df.resample('D', on='date').mean().reset_index()
    dat['cumprofit'] = dat['profit']
    # replace nan with 0 in cumprofit column
    dat['cumprofit'].fillna(0, inplace=True)
    # multiply cumprofit with 100
    dat['cumprofit'] = dat['cumprofit']
    dat['cumprofit'] = dat['cumprofit'].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dat['date'], y=dat['cumprofit'], name='channel'))
    # fig = px.line(dat, x='date', y='cumprofit', title='Profit/Loss Comparison with '+index_type)
    # find start date in dat date
    start_date = dat['date'].min()
    # find end date in dat date
    end_date = dat['date'].max()
    nseticker = yq.Ticker(stock_codes[index_type])
    ndf = nseticker.history(start=start_date, end=end_date)
    ndf['adjclose'] = (ndf['adjclose']-ndf['open'])/(ndf['open'])
    ndf['adjclose'] = ndf['adjclose']*100
    ndf['adjclose'] = ndf['adjclose'].cumsum()
    ndf.reset_index(inplace=True)
    ndf.set_index('date', inplace=True)
    fig.add_traces(go.Scatter(x=ndf.index, y=ndf['adjclose'], name=index_type))

    st.plotly_chart(fig)


row1_spacer1, row1_1, row1_spacer2 = st.columns((.2, 7.1, .2))
with row1_1:
    if uploaded_file is None:
        st.subheader('Tips Analysis for stoxMaster channel')
    else:
        st.subheader('Tips Analysis for uploaded file')


# upload csv file


df['count1'] = 1


def profit_if_start_100():

    df['date'] = pd.to_datetime(df['date'])
    dat = df
    dat = df.resample('D', on='date').mean().reset_index()
    dat['cumprofit'] = dat['profit']
    # replace nan with 0 in cumprofit column
    dat['cumprofit'].fillna(0, inplace=True)
    # multiply cumprofit with 100
    dat['cumprofit'] = dat['cumprofit']
    # dat['cumprofit'] = dat['cumprofit'].cumsum()
    dat['cumprofit'][0] = (1+dat['cumprofit'][0]/100)*100
    for i in range(1, len(dat)):
        dat['cumprofit'][i] = (dat['cumprofit'][i]/100 + 1)*dat['cumprofit'][i-1]

    return dat['cumprofit'].iloc[-1]


row2_spacer1, row2_1, row2_spacer2, row2_2, row2_spacer3 = st.columns((.2, 2.3, .4, 4.4, .2))
with row2_1:
    st.markdown('Tip tracker is a tool that enables you to analyze all the tips given by a telegram channel during its history, compare probable earnings when following those tips to simply investing in well known indexes and give you the answer to the question; scam or not? If you would like to analyze tips from some other platforms/channels, toggle the custom dataset option and upload your data.')
    # put colour of str(round(profit_if_start_100(), 2)) as red if less than 100 or green if more than 100 and write a markdown
    if profit_if_start_100() < 100:
        st.markdown('If started by 100 rupees and invested in all the tips, you would have total of '+'**:red['+str(round(profit_if_start_100(), 2))+']**'+' rupees by now.')
    else:
        st.markdown('If started by 100 rupees and invested in all the tips, you would have total of '+'**:green['+str(round(profit_if_start_100(), 2))+']**'+' rupees by now.')
    # st.markdown('If started by 100 rupees and invested in all the tips, you would have total of '+str(round(profit_if_start_100(), 2))+' rupees by now.')
with row2_2:
    draw_pie_chart()

row3_spacer1, row3_1, row3_spacer2, row3_2, row3_spacer3 = st.columns((.2, 2.3, .4, 4.4, .2))

with row3_1:
    st.markdown('No of tips')
    st.markdown('Choose a time frame to see how many tips were given in that time frame. For example, if you choose monthly, you will see how many tips were given in each month. If you choose daily, you will see how many tips were given in each day. If you choose weekly, you will see how many tips were given in each week. You can also choose to see only buy tips, sell tips or both.')
    timeframe_type = st.selectbox("Time Frame", ['monthly', 'daily', 'weekly'], key='time_frame')
    tip_type = st.selectbox("Tip Type", ["Buy", "Sell", "Both"], key='tip_type')

with row3_2:
    draw_no_tips_chart(timeframe_type, tip_type)


row4_spacer1, row4_1, row4_spacer2, row4_2, row4_spacer3 = st.columns((.2, 2.3, .4, 4.4, .2))
with row4_1:
    st.markdown('Compare Profit Index')
    st.markdown('Choose an index to compare your profit with to see if you would have made more money by simply investing in that index or not. Here we are considering that you put 100 rupees in every trade and invested in all the tips.')
    index_type = st.selectbox("Index Type", ['NIFTY50', 'SENSEX', 'RELIANCE', 'ADANI ENTERPRISES'], key='index_type')
with row4_2:
    compare_profit_index(index_type)


row5_spacer1, row5_1, row5_spacer2, row5_2, row5_spacer3 = st.columns((.2, 4.4, .4, 4.4, .2))
with row5_1:
    fig = px.histogram(df, x="sector", y="count1", histfunc="sum")
    fig.update_layout(
        xaxis_title="sector",
        yaxis_title="no of tips",
        title_text="sector wise tips"
    )
    st.plotly_chart(fig)


def issector(sec):
    if sec == "":
        return False
    return True


with row5_2:
    df2 = df[df['sector'].apply(issector)]
    tot = df2.sum()
    tot = tot['profit']
    res = (((df2.groupby('sector')['profit'].sum())/tot)*100).to_frame()
    res.reset_index(inplace=True)
    fig = px.bar(res, x='sector', y='profit', title='sector wise profit')
    st.plotly_chart(fig)


def compare_profit_index2(index_type):

    df['date'] = pd.to_datetime(df['date'])
    dat = df
    dat = df.resample('D', on='date').mean().reset_index()
    dat['cumprofit'] = dat['profit']
    # replace nan with 0 in cumprofit column
    dat['cumprofit'].fillna(0, inplace=True)
    # multiply cumprofit with 100
    dat['cumprofit'] = dat['cumprofit']
    # dat['cumprofit'] = dat['cumprofit'].cumsum()
    dat['cumprofit'][0] = (1+dat['cumprofit'][0]/100)*100
    for i in range(1, len(dat)):
        dat['cumprofit'][i] = (dat['cumprofit'][i]/100 + 1)*dat['cumprofit'][i-1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dat['date'], y=dat['cumprofit'], name='channel'))
    # fig = px.line(dat, x='date', y='cumprofit', title='Profit/Loss Comparison with '+index_type)
    # add legend in above fig
    # find start date in dat date
    start_date = dat['date'].min()
    # find end date in dat date
    end_date = dat['date'].max()
    nseticker = yq.Ticker(stock_codes[index_type])
    ndf = nseticker.history(start=start_date, end=end_date)
    ndf['adjclose'] = ((ndf['adjclose'])/(ndf['adjclose'][0]))*100
    ndf.reset_index(inplace=True)
    ndf.set_index('date', inplace=True)
    fig.add_traces(go.Scatter(x=ndf.index, y=ndf['adjclose'], name=index_type))

    st.plotly_chart(fig)


row_6_spacer1, row_6_1, row_6_spacer2, row_6_2, row_6_spacer3 = st.columns((.2, 2.3, .4, 4.4, .2))
with row_6_1:
    st.markdown('Compare Profit Index when started with 100 rupees')
    st.markdown('Choose an index to compare your profit with to see if you would have made more money by simply investing in that index or not. Here we are considering you started with 100 rupees and invested in all the tips.')
    index_type = st.selectbox("Index Type", ['NIFTY50', 'SENSEX', 'RELIANCE', 'ADANI ENTERPRISES'], key='index_type2')
with row_6_2:
    compare_profit_index2(index_type)
