import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

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
    print("Today's Date : ",today_str)
    print("Start Date : ",start_str)
    
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
    res = requests.get("http://csitrader.io/ticker_list")
    if res.status_code == 200:
        tickers = json.loads(res.text)["active_tickers"]
    else:
        tickers = ['JNK', 'LQD', 'CWB', 'ITA', 'HYG', 'PBS', 'PSP', 'PRF', 'IWS', 'RSP', 'XLB', 'IWP', 'IAI', 'IWR', 'SPXL', 'ITOT', 'IHI', 'PRFZ', 'JKE', 'RPV', 'XLG']

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

    return quadrant_df, tickers

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

def update_quadrant_df(quadrant_df, ticker_dict, selected_values):
    for ticker in selected_values:
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
    
    return quadrant_df, ticker_dict

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
        
def update_csi_df(csi_df, selected_values, start_str, today_str):
    
    for ticker in selected_values:
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
                    
    return csi_df                    



def create_csi_chart(csi_df, selected_values, start_str, today_str):

    csi_df = update_csi_df(csi_df, selected_values, start_str, today_str)
    
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

    return fig, csi_df.to_json()
   
def create_etfs_chart(quadrant_df, ticker_dict, selected_values, lookback, show_markers):

    quadrant_df, ticker_dict = update_quadrant_df(quadrant_df, ticker_dict, selected_values)

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

        fig.add_trace(go.Scatter(x=x,y=y,
                                 customdata=[(ticker, ticker_dict[ticker])]*len(x),
                                 hovertemplate="Ticker: %{customdata[0]}<br>"+\
                                               "Name: %{customdata[1]}<br>"+\
                                               "X: %{x: .2f}<br>Y: %{y: .2f}",
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
    #return fig, quadrant_df.to_json(date_format='iso', orient='split'), json.dumps(ticker_dict)
    return fig, quadrant_df.to_json(), json.dumps(ticker_dict)

csi_graph, etf_graph, multi_select_csi, multi_select_etf, etf_label, csi_label, slider_div, toggle_button, etf_lookback_slider, slider_label, start_str, today_str = [None] * 12
    
def create_layout():
    global csi_graph, etf_graph, multi_select_csi, multi_select_etf, etf_label, csi_label, slider_div,\
     toggle_button, etf_lookback_slider, slider_label, start_str, today_str
    
    ticker_dict = {
    'ITA' : 'Aerospace & Defense',
    'QQQ' : 'Nasdaq 100',
    'KO' : 'Coca-Cola Corporation',
    'BLK' : 'Blackrock',
    'ALL' : 'Allstate'
    }
    
    #csi_df = pd.read_csv("csi_and_sector_data.csv", index_col=0, parse_dates=True)
    csi_df, start_str, today_str = create_csi_df()
    #today_str, start_str = "20191231", "20180101"
    #pd.read_csv('quadrant_data.csv', index_col=0, parse_dates=True)

    quadrant_df, tickers = create_quadrant_dataframe(ticker_dict)
    
    
    
    
    slider_value = 8
    csi_selects = csi_df.columns[:2]

    options_csi = []
    for key, val in csi_mapping.items():
        options_csi.append({"label": val, "value": key })

    options_etf = []
    for ticker in tickers:
        options_etf.append({"label": ticker, "value": ticker })
        

    # Added "config={"displayModeBar": False}" to hide the displayModeBar from the user. For this graph, I feel that onlyl showing the range slider is sufficient.
    csi_graph = dcc.Graph(figure=create_csi_chart(csi_df, csi_df.columns[:2], start_str, today_str)[0], id="csi_chart", className="figure", style={"height":"70vh",}, config={"displayModeBar": False})

    etf_graph = dcc.Graph(figure=create_etfs_chart(quadrant_df, ticker_dict, list(ticker_dict.keys())[0], slider_value, True), id="etf_chart", className="figure", style={"height":"65vh",})


    left_pan_radio = dcc.RadioItems(options = [{'label': 'CSI Chart', 'value': 'CSI'},
                                               {'label': 'ETFs Chart', 'value': 'ETF'}], value="CSI", id="chart-selection", style={"color":"white", "font-size":20})

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


    right_pane_children = html.Div(children=[csi_label, multi_select_csi, csi_graph])
    right_pan = html.Div(id="right-pane", children=[right_pane_children], className="nine columns", style={'backgroundColor': colors['background'], "padding":"0px", "margin":"0px"})

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

    left_pan = html.Div(children=[header2, radio_div], className="three columns", style={'backgroundColor': colors['background']})


    slider_div = html.Div(children=[slider_label, etf_lookback_slider], className="eleven columns")


    main_dashboard = children=html.Div(children=[left_pan, right_pan], className="row")


    csi1 = html.Div(id='csi1', children=csi_df.to_json() , style={'display': 'none'})
    csi_store = dcc.Store(id='csi', data=csi_df.to_dict())
    etf1 = html.Div(id='etf1', children=quadrant_df.to_json(), style={'display': 'none'})
    etf_store = dcc.Store(id='etf', data=quadrant_df.to_dict())
    ticker_data1 = html.Div(id='ticker-dict1', children=json.dumps(ticker_dict), style={'display': 'none'})
    ticker_store = dcc.Store(id='ticker-dict', data=ticker_dict)
    
    layout = html.Div(style={'backgroundColor': colors['background'], "height":"120vh"},
                            children=[header, main_dashboard, csi1, csi_store, etf1, etf_store, ticker_data1, ticker_store], className="row")
                            
    return layout

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__,  suppress_callback_exceptions=True,
                    external_stylesheets=external_stylesheets,
                    meta_tags=[{"name": "viewport", "content": "width=device-width"}])
                    
app.layout = create_layout

@app.callback(Output('right-pane', 'children'), [Input('chart-selection', 'value'), Input('csi1', 'children'), Input('etf1', 'children'), Input('ticker-dict1', 'children')])
def update_figure(selected_chart, csi_data, etf_data, ticker_data):
    if selected_chart == "CSI":
        csi_df = pd.DataFrame(csi_data)
        csi_graph.figure, csi_df = create_csi_chart(csi_df, multi_select_csi.value, start_str, today_str).to_dict()

        return [csi_label, multi_select_csi, csi_graph]
    else:
        quadrant_df = pd.read_json(etf_data)
        ticker_dict = json.loads(ticker_data)
        
        etf_graph.figure, quadrant_df, ticker_dict = create_etfs_chart(quadrant_df, ticker_dict, multi_select_etf.value, etf_lookback_slider.value, toggle_button.value).to_dict()

        return [etf_label, multi_select_etf, etf_graph, html.Div(children=[slider_div, toggle_button], className="row")]


@app.callback([Output('csi_chart', 'figure'), Output('csi1', 'children')], [Input('multi_select_csi', 'value')], [State('csi1', 'children')])
def update_csi_lines_chart(selected_values, csi_data):
    csi_df = pd.read_json(csi_data)
    
    if not selected_values:
        selected_values = ["csi1", "csi2"]
        multi_select_csi.value = selected_values

    return create_csi_chart(csi_df, selected_values, start_str, today_str)
    

@app.callback([Output('etf_chart', 'figure'), Output('etf1', 'children'), Output('ticker-dict1', 'children')], [Input('multi_select_etf', 'value'), Input('lookback_slider', 'value'), Input('toggle_markers', 'value')], [State('etf1', 'children'),  State('ticker-dict1', 'children')])
def update_etf_chart(selected_values, lookback_value, show_markers, etf_data, ticker_data):
    quadrant_df = pd.read_json(etf_data)
    ticker_dict = json.loads(ticker_data)
    if not selected_values:
        selected_values = ["ITA", "QQQ", "KO", "BLK", "ALL"]
        multi_select_etf.value = selected_values

    return create_etfs_chart(quadrant_df, ticker_dict, selected_values, lookback_value, show_markers)

app.run_server(debug=True)
