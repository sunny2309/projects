import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import dash_daq as daq

import requests
import datetime
import json

csi_mapping = {
    'csi1': 'Cyclical Strength Index',
    'csi2': 'Cyclical Strength Index Lag',
    'XLC': 'Communications',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLE': 'Energy',
    'XLF': 'Financials',
    'XLV': 'Healthcare',
    'XLI': 'Industrials',
    'XLB': 'Materials',
    'XLRE': 'Real Estate',
    'XRT': 'Retail',
    'XSD': 'Semiconductors',
    'XLK': 'Technology',
    'XLU': 'Utilities',
    'UUP' : 'Dollar Index',
    'TLT' : 'Long-term Treasuries'
}

colors = {
    'background': '#21252C',
    'text': '#7FDBFF'
}

color_scales = px.colors.cyclical.HSV + px.colors.diverging.Spectral + px.colors.cyclical.mygbm
color_scales = color_scales * 3
rng  = np.random.RandomState(123)
rng.shuffle(color_scales)

def retrieve_tickers():
    tickers = ['JNK', 'LQD', 'CWB', 'ITA', 'HYG', 'PBS', 'PSP', 'PRF', 'IWS', 'RSP', 'XLB', 'IWP', 'IAI', 'IWR', 'SPXL', 'ITOT', 'IHI', 'PRFZ', 'JKE', 'RPV', 'XLG']
    try:
        res = requests.get("http://csitrader.io/ticker_list")
        if res.status_code == 200:
            tickers = json.loads(res.text)["active_tickers"]
    except:
        print("Failed to retrieve tickers. We'll try one more time.")
        
        try:
            res = requests.get("http://csitrader.io/ticker_list")
            if res.status_code == 200:
                tickers = json.loads(res.text)["active_tickers"]
        except:
            print("Failed to retrieve tickers again. We'll now use default tickers.")
            
    return tickers

def retrieve_csi_main_indexes(url):
    csi_df = None
    try:
        res = requests.get(url)
        if res.status_code == 200:
            csi_df = pd.DataFrame(json.loads(json.loads(res.text)))
            csi_df.index = pd.to_datetime([datetime.datetime.fromtimestamp(int(dt[:-3])) for dt in csi_df.index])
    except Exception as e:
        print(e)

    return csi_df
    
def create_csi_df():
    today = datetime.datetime.today()
    start = today - datetime.timedelta(days=1095)
    today_str = datetime.datetime.strftime(today, "%Y%m%d")        #"20191231" #"%d%d%d"%(today.year, today.month, today.day)
    start_str = datetime.datetime.strftime(start, "%Y%m%d")        #"20180101" # "%d%d%d"%(start.year, start.month, start.day)
    #print("Today's Date : ",today_str)
    #print("Start Date : ",start_str)
    
    url = "http://csitrader.io/csi/parameters?start_date=%s&end_date=%s"%(start_str, today_str)
    
    try:
        csi_df = retrieve_csi_main_indexes(url)
        if not isinstance(csi_df, pd.DataFrame):
            raise Exception("Failed to Retrieve data")
    except Exception as e:
        print("Failed to retrieven CSI Basic Indexes first time. Trying Again.")
        
        try:
            csi_df = retrieve_csi_main_indexes(url)
        except Exception as e:
            print("Failed to retrieve CSI Basic Indexes 2nd Time.")    
    
    csi_df = csi_df.fillna(value=0)
    
    return csi_df, start_str, today_str

def retrieve_quadrant_data(url, ticker):
    ticker_df = None
    try:
        res = requests.get(url)
        if res.status_code == 200:
            ticker_df = pd.DataFrame(json.loads(json.loads(res.text)))
            ticker_df.index = pd.to_datetime([datetime.datetime.fromtimestamp(int(dt[:-3])) for dt in ticker_df.index])
    except Exception as e:
        print("Failed to retreive ticker data")
    
    return ticker_df
    
def create_quadrant_dataframe(ticker_dict):
    quadrant_df = None
    for ticker in ticker_dict.keys():
        try:
            url = "http://csitrader.io/quadrant/parameters?ticker=%s"%ticker
            ticker_df = retrieve_quadrant_data(url, ticker)
            if not isinstance(ticker_df, pd.DataFrame):
                raise Exception("Ticker %s failed to retrieve quadrant data"%ticker)
                
            quadrant_df = ticker_df.copy() if not isinstance(quadrant_df,pd.DataFrame) else quadrant_df.join(ticker_df)
        except:
            print("Ticker %s failed to retrieve quadrant data. We'll try one more time."%ticker)
            
            try:
                ticker_df = retrieve_quadrant_data(url, ticker)
                quadrant_df = ticker_df.copy() if not isinstance(quadrant_df,pd.DataFrame) else quadrant_df.join(ticker_df)
            except:
                print("Ticker %s failed to retrieve quadrant data.No More try."%ticker)
                
    quadrant_df = quadrant_df.fillna(value=0)

    return quadrant_df

def retrieve_quadrant_performance(ticker, earlier_date=None, current_date=None, method=None, relative_to_snp=None):
    perf_data  = {"SUM_Q1":0, "SUM_Q2":0, "SUM_Q3":0, "SUM_Q4":0}
    ear_d, curr_d = earlier_date.replace("-",""), current_date.replace("-","")
    try:
        res = requests.get("http://csitrader.io/performance/parameters?ticker=%s&value=%s&start_date=%s&end_date=%s&customindex=%s"%(ticker, method, ear_d, curr_d, str(relative_to_snp)))
        if res.status_code == 200:
            perf_data = json.loads(res.text)
    except Exception as e:
        print("Failed to Retrieve Ticker : %s data. Will try one more time."%ticker)
        
        try:
            res = requests.get("http://csitrader.io/performance/parameters?ticker=%s&value=%s&start_date=%s&end_date=%s&customindex=%s"%(ticker, method, ear_d, curr_d, str(relative_to_snp)))
            if res.status_code == 200:
                perf_data = json.loads(res.text)
        except:
            print("Failed to retrieve ticker data one more time. Will not try now")
            
    return perf_data
                
def create_quadrant_bar_data(ticker_dict2):
    quadrant_bar_data  = []
    for ticker in ticker_dict2.keys():
        perf_data = retrieve_quadrant_performance(ticker, earlier_date, current_date, method, relative_to_snp)
        quadrant_bar_data.append(perf_data) 
       
    quadrant_bar_df = pd.DataFrame(quadrant_bar_data)
    quadrant_bar_df = quadrant_bar_df * 100
    quadrant_bar_df.index = ticker_dict2.keys()

    return quadrant_bar_df
    
    
def retrieve_company_name_from_ticker(ticker):
    company_name = "UNKNOWN"
    try:
        url = "http://csitrader.io/company_name/parameters?ticker=%s"%ticker
        res = requests.get(url)
        if res.status_code == 200:
            company_name_res = json.loads(res.text)
            if isinstance(company_name_res["company_name"], str):
                company_name = str(company_name_res["company_name"])
            else:
                raise Exception("Failed to retrieve company name for ticker : %s"%ticker)
        else:
            raise Exception("Failed to retrieve company name for ticker : %s"%ticker)
    except:
        print("Failed to retrieve company name for ticker : %s. Trying one more time. "%ticker)
        
        try:
            res = requests.get(url)
            company_name_res = json.loads(res.text)
            if isinstance(company_name_res["company_name"], str):
                company_name = str(company_name_res["company_name"])
        except:
            print("Failed to retrieve company name for ticker : %s. Failed 2nd Time."%ticker)
            
    return company_name

def update_quadrant_df(selected_tickers):
    global quadrant_df, ticker_dict    
    
    
    for ticker in selected_tickers:
        col_name = ticker + "_value1"
        if ticker not in ticker_dict:
            company_name = retrieve_company_name_from_ticker(ticker)
            ticker_dict.update({ticker:company_name})
    
        if col_name not in quadrant_df.columns:
            try:
                url = "http://csitrader.io/quadrant/parameters?ticker=%s"%ticker
                ticker_df = retrieve_quadrant_data(url, ticker)
                
                if not isinstance(ticker_df, pd.DataFrame):
                    raise Exception("Ticker %s failed to retrieve quadrant data"%ticker)
                
                quadrant_df = quadrant_df.join(ticker_df)
            except Exception as e:
                print("Ticker %s failed to retrieve quadrant data. We'll now try one more time."%ticker)
                
                try:
                    ticker_df = retrieve_quadrant_data(url, ticker)
                    quadrant_df = quadrant_df.join(ticker_df)
                except:
                    print("Ticker %s failed to retrieve quadrant data one more time."%ticker)


def retrieve_csi_data(url, ticker):
    ticker_df = None
    try:
        res = requests.get(url)
        if res.status_code == 200:
            ticker_df = pd.DataFrame(json.loads(json.loads(res.text)))
            ticker_df.index = pd.to_datetime([datetime.datetime.fromtimestamp(int(dt[:-3])) for dt in ticker_df.index])
            ticker_df = ticker_df.drop(columns=["value2"])
            ticker_df.columns  = [ticker]
    except:
        print("Failed to retrieve data for ticker : %s"%ticker)
    
    return ticker_df
        
def update_csi_df(selected_tickers, start_str, today_str):
    global csi_df
    
    for ticker in selected_tickers:
        if ticker not in csi_df.columns:
            try:
                url = "http://csitrader.io/qqe_ci/parameters?ticker=%s&start_date=%s&end_date=%s"%(ticker, start_str, today_str)
                #print(url)
                ticker_df = retrieve_csi_data(url, ticker)
                
                if not isinstance(ticker_df, pd.DataFrame):
                    raise Exception("Ticker %s failed to retrieve csi data."%ticker)
                    
                csi_df = csi_df.join(ticker_df)
            except Exception as e:
                print(e)
                
                try:
                    ticker_df = retrieve_csi_data(url, ticker)                    
                    csi_df = csi_df.join(ticker_df)
                except Exception as e:
                    print("Ticker %s failed to retrieve csi data last time"%ticker)

def update_quadrant_bar_df(selected_tickers, earlier_date_p, current_date_p, method_p, relative_to_snp_p):
    global quadrant_bar_df, ticker_dict2, earlier_date, current_date, method, relative_to_snp
    
    if (earlier_date!= earlier_date_p) or (current_date!=current_date_p) or (method!=method_p) or (relative_to_snp!=relative_to_snp_p):
        quadrant_bar_df, earlier_date, current_date, method, relative_to_snp  = None, earlier_date_p, current_date_p, method_p, relative_to_snp_p
        for ticker in selected_tickers:
            if ticker not in ticker_dict2:
                company_name = retrieve_company_name_from_ticker(ticker)
                ticker_dict2.update({ticker:company_name})
        
            quadrant_data = retrieve_quadrant_performance(ticker, earlier_date, current_date, method, relative_to_snp)
            ticker_df = pd.DataFrame([quadrant_data], index=[ticker])
            if method in ["SUM", "AVG"]:
                ticker_df =  ticker_df *100
            quadrant_bar_df = pd.concat([quadrant_bar_df, ticker_df]) if isinstance(quadrant_bar_df, pd.DataFrame) else ticker_df.copy()
            
    else:
        for ticker in selected_tickers:
            if ticker not in ticker_dict2:
                company_name = retrieve_company_name_from_ticker(ticker)
                ticker_dict2.update({ticker:company_name})
        
            if ticker not in quadrant_bar_df.index:
                quadrant_data = retrieve_quadrant_performance(ticker, earlier_date, current_date, method, relative_to_snp)
                ticker_df = pd.DataFrame([quadrant_data], index=[ticker])
                if method in ["SUM", "AVG"]:
                    ticker_df =  ticker_df *100
                quadrant_bar_df = pd.concat([quadrant_bar_df, ticker_df])
                
    quadrant_bar_df = quadrant_bar_df.fillna(0)


def create_csi_chart(selected_values, start_str, today_str):

    update_csi_df(selected_values, start_str, today_str)
    
    fig = go.Figure()

    charts = []
    for idx, col in enumerate(selected_values):
        if col not in csi_df.columns:
            continue
            
        ticker_industry_name = csi_mapping.get(col.upper()) if csi_mapping.get(col.upper(), None) else col.upper()
        scat =  go.Scatter(x=csi_df.index, y=csi_df[col], name=col.upper(),
                        line = dict(color=color_scales[idx]),
                        hovertemplate="Ticker: %s<br>"%(col.upper())+\
                                      "Name : %s<br>"%(ticker_industry_name)+\
                                      "X: %{x}<br>Y: %{y: .2f}"

                       )
        charts.append(scat)


    fig = go.Figure(data=charts)



    fig.update_layout(title_text="Cyclical Strength Index",
                      xaxis_title='Date Range',
                      yaxis_title='Index',
                      legend_title_text="Ticker",
                      template="plotly_dark",
                      #paper_bgcolor="#21252C", plot_bgcolor="#21252C",
                      autosize=True,
                      margin=dict(t=80, b=50),
                      )

    fig.update_xaxes(rangeslider_visible=True,

                     rangeselector=dict(
                                     buttons=[
                                     dict(count=1, label="1m", step="month", stepmode="backward"),
                                     dict(count=3, label="3m", step="month", stepmode="backward"),
                                     dict(count=6, label="6m", step="month", stepmode="backward"),
                                     dict(count=1, label="YTD", step="year", stepmode="todate"),
                                     dict(count=1, label="1y", step="year", stepmode="backward"),
                                     dict(step="all")
                                    ],
                                     font = dict(size=11.5, color='#000000')
                                )
                    )

    # Set initial "zoom" range of the slider
    zoom_end_dt = datetime.datetime.today()
    zoom_start_dt = datetime.datetime.today() - datetime.timedelta(days=180)

    zoom_end_str = datetime.datetime.strftime(zoom_end_dt, "%Y-%m-%d")
    zoom_start_str = datetime.datetime.strftime(zoom_start_dt, "%Y-%m-%d")

    initial_range = [
        zoom_start_str, zoom_end_str
    ]

    fig['layout']['xaxis'].update(range=initial_range)

    # Change default "hover" effect to "Compare data on hover"
    fig.update_layout(hovermode='x')

    return fig
   
def create_etfs_chart(selected_values, lookback, show_markers):

    update_quadrant_df(selected_values)

    #selected_values = [val for val in selected_values if val+"_value1" in quadrant_df.columns]

    quadrant_df_sub = quadrant_df.iloc[-lookback:]
    fig = go.Figure()

    xs, ys = [], []
    annotations = []

    for idx, ticker in enumerate(selected_values):
        col1, col2 = ticker+'_value1', ticker+'_gmean'
        
        if col1 not in quadrant_df_sub.columns:
            continue
        
        x = quadrant_df_sub[col1].values.tolist()
        y = quadrant_df_sub[col2].values.tolist()
        xs.extend(x)
        ys.extend(y)
        
               
        custom_data = [(ticker, ticker_dict[ticker], "%d-%d-%d"%(dt.year,dt.month, dt.day), velocity) for dt,velocity in zip(quadrant_df_sub.index, quadrant_df_sub[ticker+"_velocity"])]
        
        fig.add_trace(go.Scatter(x=x,y=y,
                                 customdata=custom_data,
                                 hovertemplate="Ticker: %{customdata[0]}<br>"+\
                                               "Name: %{customdata[1]}<br>"+\
                                               "X: %{x: .2f}<br>Y: %{y: .2f}<br>Z: %{customdata[3]: .2f}<br>"+\
                                               "Date: %{customdata[2]}",
                                 name=ticker, #ticker_dict[ticker],
                                 mode='lines+markers',
                                 marker=dict(size=12,  color=color_scales[idx]),
                                 line=dict(width=1.5, color=color_scales[idx])))
        if show_markers:
            annotations.append(go.layout.Annotation(
                               x= x[-1], y= y[-1], ## X, Y Ends
                               ax= x[-1],  ay= y[-1], ## X, Y Starts
                               xref="x", yref="y",
                               axref = "x", ayref='y',
                               text=ticker,
                               font=dict(
                               family="Courier New, monospace",
                               size=16,
                               color="#ffffff"
                               ),
                               align="left",
                               xshift=20, #yshift=10
                               #bgcolor="grey"
                             ))

#        fig.update_traces(textposition='outside')
        '''
        for i in range(len(x)-1):
            if i < 2:
                fig.add_annotation(x= x[i+1], y= y[i+1], ## X, Y Ends
                                   ax= x[i],  ay= y[i], ## X, Y Starts
                                   #xref="x", yref="y",
                                   axref = "x", ayref='y',
                                   text="",
                                   showarrow=True,
                                   arrowhead = 2,
                                   arrowwidth=1.5,)
            else:
                annotations.append(go.layout.Annotation(
                                   x= x[i+1], y= y[i+1], ## X, Y Ends
                                   ax= x[i],  ay= y[i], ## X, Y Starts
                                   xref="x", yref="y",
                                   axref = "x", ayref='y',
                                   text="",
                                   showarrow=True,
                                   arrowhead = 2,
                                   arrowwidth=1.5,
                                   arrowcolor="white",))
        '''

    fig.add_shape(
                    type="rect",
                    x0=min(xs)-3,
                    y0=min(ys)-3,
                    x1=0,
                    y1=100,
                    fillcolor="red", opacity=0.3,
                    line=dict(width=0),
    )

    fig.add_shape(
                    type="rect",
                    x0=0,
                    y0=100,
                    x1=max(xs)+3,
                    y1=max(ys)+3,
                    fillcolor="green", opacity=0.3,
                    line=dict(width=0),
    )

    fig.add_shape(
                    type="rect",
                    x0=min(xs)-3,
                    y0=100,
                    x1=0,
                    y1=max(ys)+3,
                    fillcolor="yellow", opacity=0.3,
                    line=dict(width=0),
    )

    fig.add_shape(
                    type="rect",
                    x0=0,
                    y0=min(ys)-3,
                    x1=max(xs)+3,
                    y1=100,
                    fillcolor="yellow", opacity=0.3,
                    line=dict(width=0),
    )

    fig.add_annotation(x=min(xs)-1.5, y=max(ys)+1.5, text="Reflation", showarrow=False, font_size=15, opacity=0.7)
    fig.add_annotation(x=min(xs)-1.5, y=min(ys)-1.5, text="Contraction", showarrow=False, font_size=15, opacity=0.7)
    fig.add_annotation(x=max(xs)+1.5, y=max(ys)+1.5, text="Expansion", showarrow=False, font_size=15, opacity=0.7)
    # Changed text to "Consolidation"
    fig.add_annotation(x=max(xs)+1.5, y=min(ys)-1.5, text="Consolidation", showarrow=False, font_size=15, opacity=0.7)

    fig["layout"]["annotations"] += tuple(annotations)

    fig.update_layout(title="Historical Best ETFs",
                      xaxis_title="Strength vs S&P 500",
                      yaxis_title="Momentum of Strength",
                      template="plotly_dark",
                      legend_title_text="Ticker",
                      #legend_orientation="h",
                      #grid=dict(xgrid=None, ygrid=None)
                      #showlegend=False,
                      hoverlabel=dict(
                            bgcolor="white",
                            font_size=14,
                            font_family="Rockwell"
                            ),
                      #paper_bgcolor="#21252C", plot_bgcolor="#21252C",
                      autosize=True,
                      margin=dict(t=50, b=10),
                      )
    return fig


def create_quadrant_performance_bar_chart(selected_tickers, earlier_date, current_date, method, relative_to_snp):
    update_quadrant_bar_df(selected_tickers, earlier_date, current_date, method, relative_to_snp)
    
    quadrant_bar_df_sub = quadrant_bar_df.loc[selected_tickers]
    
    data = []
    for idx, (ticker, row) in enumerate(zip(quadrant_bar_df_sub.index, quadrant_bar_df_sub.values)):
        data.append(go.Bar(x=["Quadrant1", "Quadrant2", "Quadrant3", "Quadrant4"], y=row, 
                           name=ticker,
                           customdata= [(ticker, ticker_dict2[ticker])]*4,
                           hovertemplate="Ticker: %{customdata[0]}<br>"+\
                                                   "Name: %{customdata[1]}<br>"+\
                                                   "X: %{x}<br>Y: %{y: .2f}%",
                           marker_color=color_scales[idx],
                           text=["%.2f%%"%r for r in row],
                           textposition='outside'
                          ))
        
    fig = go.Figure(data=data)

    fig.update_layout(title="10-YR Performance Chart",
                          xaxis_title="Quadrants",
                          yaxis_title="Total Cummulative Returns (%)",
                          #template="plotly_dark",
                          #grid=dict(xgrid=None, ygrid=None)
                          #showlegend=False,
                          template="plotly_dark",
                          legend_title_text="Ticker",
                          hoverlabel=dict(
                                bgcolor="white", 
                                font_size=14, 
                                font_family="Rockwell"
                                ),
                          barmode='group',
                          autosize=True,
                          margin=dict(t=50, b=10),
                          )

    fig.update_yaxes(ticksuffix="%")    
    
    return fig
    
    
csi_graph, etf_graph, multi_select_csi, multi_select_etf, csi_df, etf_label, csi_label, slider_div, toggle_button, etf_lookback_slider, slider_label, quadrant_df, ticker_dict = [None] * 13
start_str, today_str, quadrant_bar_df, ticker_dict2, performance_bar_graph, perf_label, multi_select_perf, tickers, perf_extra_widgets  = [None] * 9
earlier_date, current_date, method, relative_to_snp = None, None, "SUM", False
    
def create_layout():
    global csi_graph, etf_graph, multi_select_csi, multi_select_etf, csi_df, etf_label, csi_label, slider_div,\
     toggle_button, etf_lookback_slider, slider_label, quadrant_df, ticker_dict, start_str, today_str, quadrant_bar_df, ticker_dict2,\
     performance_bar_graph, perf_label, multi_select_perf, tickers, perf_extra_widgets, earlier_date, current_date, method, relative_to_snp
    
    ticker_dict = {'ITA' : 'Aerospace & Defense', 'QQQ' : 'Nasdaq 100', 'KO' : 'Coca-Cola Corporation', 'BLK' : 'Blackrock', 'ALL' : 'Allstate'}
    
    ticker_dict2 = {"IWM" :"Russell 200", "SPY":"S&P 500", "LQD":"Corporate Bonds", "TLT":"20+ YR Treasuries"}
    
    current_date = datetime.datetime.now()
    earlier_date = current_date - datetime.timedelta(days=3650)
    
    current_date = str(current_date.date())
    earlier_date = str(earlier_date.date())
    #csi_df = pd.read_csv("csi_and_sector_data.csv", index_col=0, parse_dates=True)
    if not isinstance(csi_df, pd.DataFrame):
        csi_df, start_str, today_str = create_csi_df()
    #today_str, start_str = "20191231", "20180101"
    #pd.read_csv('quadrant_data.csv', index_col=0, parse_dates=True)
    if not tickers:
        tickers  = retrieve_tickers()
        tickers = tickers + list(ticker_dict2.keys())
        
    if not isinstance(quadrant_df, pd.DataFrame):
        quadrant_df = create_quadrant_dataframe(ticker_dict)
    
    if not isinstance(quadrant_bar_df, pd.DataFrame):
        quadrant_bar_df = create_quadrant_bar_data(ticker_dict2)
    
    
    
    slider_value = 8
    csi_selects = csi_df.columns[:2]

    options_csi = []
    for key, val in csi_mapping.items():
        options_csi.append({"label": val, "value": key })

    options_etf = []
    for ticker in sorted(tickers):
        options_etf.append({"label": ticker, "value": ticker })
        
    # Added "config={"displayModeBar": False}" to hide the displayModeBar from the user. For this graph, I feel that onlyl showing the range slider is sufficient.
    csi_graph = dcc.Graph(figure=create_csi_chart(csi_df.columns[:2], start_str, today_str), id="csi_chart", className="figure", style={"height":"70vh",}, config={"displayModeBar": False})

    etf_graph = dcc.Graph(figure=None, id="etf_chart", className="figure", style={"height":"65vh",})
    
    performance_bar_graph = dcc.Graph(figure=None, id="performance_bar_chart", className="figure", style={"height":"70vh"})


    left_pan_radio = dcc.RadioItems(options = [{'label': 'CSI Chart', 'value': 'CSI'},
                                               {'label': 'ETFs Chart', 'value': 'ETF'},
                                               {'label': '10 YR Performance Chart', 'value': 'PERF'},], value="CSI", id="chart-selection", style={"color":"white", "font-size":20})

    radio_div = html.Div(children=[left_pan_radio], style={"display":"flex","justifyContent":"center"})

    h1 = html.H1("William's Financial Dashboard", className="title-header", style={"color":"white", "textAlign": "center", "padding":"10px"})
    h4 = html.H4("List of Available Charts", style={"color":"white", "textAlign": "center"})

    header = html.Div(children=[h1])
    header2 = html.Div(children=[h4])

    csi_label = html.Label(children="CSI MultiSelect", style={"color":"white", "margin":"3px"})

    multi_select_csi = dcc.Dropdown(
                    id="multi_select_csi",
                    options=options_csi,
                    value=[opt["value"] for opt in options_csi][:2],
                    multi=True,
                    persistence=True, persistence_type="session",
                    clearable=False,
                    # Adding placeholder text for style purposes
                    placeholder = "Add Ticker to Graph",
                    style={"background-color": "#21252C", "padding":"2px","margin":"2px 0px 3px 0px", "select-value-color":"red"}
                )

    etf_label = html.Label(children="ETFs MultiSelect", style={"color":"white", "margin":"3px"})

    multi_select_etf = dcc.Dropdown(
                    id="multi_select_etf",
                    options=options_etf,
                    value=list(ticker_dict.keys()), #[opt["value"] for opt in options_etf],
                    multi=True,
                    persistence=True, persistence_type="session",
                    clearable=False,
                    style={"background-color": "#21252C", "padding":"2px","margin":"2px 0px 3px 0px"}
                )
    
    slider_label = html.Label(children="Lookback Window", style={"color":"white", "padding":"3px"})

    etf_lookback_slider = dcc.Slider(
                                    id="lookback_slider",
                                    min=1,
                                    max=quadrant_df.shape[0],
                                    marks={i: str(i) for i in range(1, quadrant_df.shape[0]+1)},
                                    value=8,
                                    persistence=True, persistence_type="session",
                                    )
    
    toggle_button = daq.ToggleSwitch(
                                    id="toggle_markers",
                                    label='Markers',
                                    labelPosition='top',
                                    theme="dark",
                                    color="grey",
                                    style={"color":"white"},
                                    value=True,
                                    size=50,
                                    persistence=True, persistence_type="session"
                                    )
    
    perf_label = html.Label(children="10-YR Performance MultiSelect", style={"color":"white", "margin":"3px"})

    multi_select_perf = dcc.Dropdown(
                    id="multi_select_perf",
                    options=options_etf,
                    value=list(ticker_dict2.keys()), #[opt["value"] for opt in options_etf],
                    multi=True,
                    persistence=True, persistence_type="session",
                    clearable=False,
                    style={"background-color": "#21252C", "padding":"2px","margin":"2px 0px 3px 0px"}
                )
    

    date_range_picker_earlier = html.Div(children=[html.Label(children="Start Date", style={"color":"white", "margin":"0px"}),
                                                              html.Div(children=dcc.DatePickerSingle(id = "date_range_earlier", date = earlier_date), 
                                                                        style={"background-color": "#21252C", "color":"navy"})],
                                                                                   className="two columns", 
                                                                                   style={"margin":"0px 0px 6px 0px", "background-color": "#21252C", "color":"dodgerblue"})
    
    date_range_picker_current = html.Div(children=[
                                            html.Label(children="End Date", style={"color":"white", "margin":"0px"}),
                                            dcc.DatePickerSingle(id = "date_range_current", date = current_date, max_date_allowed = current_date)],
                                                                 className="two columns",
                                                                 style={"margin":"0px 0px 6px 0px"})
                                                                     
    
    perf_chart_dropdown = html.Div(children=[html.Label(children="Method", style={"color":"white", "margin":"0px"}),
                                             dcc.Dropdown(options = [{"label":k, "value":k} for k in ["SUM", "SQN", "AVG", "STD"]], 
                                                   value=method,
                                                   id="perf_dropdown",
                                                   persistence=True, persistence_type="session",
                                                   clearable=False,
                                                   style={"background-color": "#21252C", "fontColor":"dodgerblue"})], className="two columns", style={"margin":"0px 0px 6px 0px"})
                                       
    toggle_button_perf = daq.ToggleSwitch(
                                id="toggle_perf",
                                label='Relative Performance\n to S&P 500',
                                labelPosition='top',
                                theme="dark",
                                color="grey",
                                style={"color":"white", "margin":"0px 5px, 6px 5px"},
                                value=False,
                                size=50,
                                persistence=True, persistence_type="session"
                                )
    
    perf_extra_widgets = html.Div(children=[date_range_picker_earlier, date_range_picker_current, perf_chart_dropdown, toggle_button_perf], className="row", style={"display":"flex"})

    right_pane_children = html.Div(children=[csi_label, multi_select_csi, csi_graph])
    right_pan = html.Div(id="right-pane", children=[right_pane_children], className="nine columns", style={'backgroundColor': colors['background'], "padding":"0px", "margin":"0px"})


    left_pan = html.Div(children=[header2, radio_div], className="three columns", style={'backgroundColor': colors['background']})


    slider_div = html.Div(children=[slider_label, etf_lookback_slider], className="eleven columns")


    main_dashboard = children=html.Div(children=[left_pan, right_pan], className="row")


    layout = html.Div(style={'backgroundColor': colors['background'], "height":"120vh"},
                            children=[header, main_dashboard], className="row") # html.Hr(style={"color":"black", "padding":"0px", "margin":"0px"})
                            
    return layout

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__,  suppress_callback_exceptions=True,
                    external_stylesheets=external_stylesheets,
                    meta_tags=[{"name": "viewport", "content": "width=device-width"}])
                    
app.layout = create_layout

@app.callback(Output('right-pane', 'children'), [Input('chart-selection', 'value')])
def update_figure(selected_chart):
    if selected_chart == "CSI":

        return [csi_label, multi_select_csi, csi_graph]
        
    elif selected_chart == "ETF":

        etf_graph.figure = create_etfs_chart(multi_select_etf.value, etf_lookback_slider.value, toggle_button.value).to_dict()

        return [etf_label, multi_select_etf, etf_graph, html.Div(children=[slider_div, toggle_button], className="row")]
    
    elif selected_chart == "PERF":
        
        performance_bar_graph.figure = create_quadrant_performance_bar_chart(multi_select_perf.value, earlier_date, current_date, method, relative_to_snp)
        
        return [perf_label, multi_select_perf, perf_extra_widgets, performance_bar_graph]


@app.callback(Output('csi_chart', 'figure'), [Input('multi_select_csi', 'value')])
def update_csi_lines_chart(selected_values):

    if not selected_values:
        selected_values = ["csi1", "csi2"]
        multi_select_csi.value = selected_values

    return create_csi_chart(selected_values, start_str, today_str)


@app.callback(Output('etf_chart', 'figure'), [Input('multi_select_etf', 'value'), Input('lookback_slider', 'value'), Input('toggle_markers', 'value')])
def update_etf_chart(selected_values, lookback_value, show_markers):

    if not selected_values:
        selected_values = ["ITA", "QQQ", "KO", "BLK", "ALL"]
        multi_select_etf.value = selected_values


    return create_etfs_chart(selected_values, lookback_value, show_markers)

@app.callback(Output('performance_bar_chart', 'figure'), [Input('multi_select_perf', 'value'), Input('date_range_earlier', 'date'), Input('date_range_current', 'date'), Input('perf_dropdown', 'value'), Input('toggle_perf', 'value')])
def update_performance_chart(selected_values, earlier_date, current_date, method, relative_to_snp):
    
    if not selected_values:
        selected_values = ["IWM", "SPY", "LQD", "TLT"]
        multi_select_perf.value = selected_values

    return create_quadrant_performance_bar_chart(selected_values, earlier_date, current_date, method, relative_to_snp)
    
app.run_server(debug=True)


