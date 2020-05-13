import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


ticker_dict = {
    'JNK' : 'Junk Bonds',
    'LQD' : 'Corporate Bonds',
    'CWB' : 'Convertible Securities',
    'ITA' : 'Aerospace & Defense',
    'HYG' : 'High Yield Bonds',
    'PBS' : 'Dynamic Media',
    'PSP' : 'Private Equity',
    'PRF' : 'FTSE 1000',
    'IWS' : 'MidCap Value',
    'RSP' : 'Eql Weight SP500',
    'XLB' : 'Materials',
    'IWP' : 'MidCap Growth',
    'IAI' : 'Broker Dealers',
    'IWR' : 'MidCap Index',
    'SPXL' : 'Levered 3x SP500',
    'ITOT' : 'SP1500',
    'IHI' : 'Medical Devices',
    'PRFZ' : 'Preferred Stock',
    'JKE' : 'LargeCap Growth',
    'RPV' : 'SP Pure Value',
    'XLG' : 'Top-50 SP500'
}

colors = {
    'background': '#21252C',
    'text': '#7FDBFF'
}

color_scales = px.colors.cyclical.HSV + px.colors.diverging.Spectral + px.colors.cyclical.mygbm
rng  = np.random.RandomState(123)
rng.shuffle(color_scales)


csi_df = pd.read_csv("csi_and_sector_data.csv", index_col=0, parse_dates=True)
quadrant_df = pd.read_csv('quadrant_data.csv', index_col=0, parse_dates=True)


slider_value = 8
etf_selects = list(ticker_dict.keys())
csi_selects = csi_df.columns[:2]

options_csi = []
for col in csi_df.columns:
    options_csi.append({"label": col.upper(), "value": col })

options_etf = []
for key, val in ticker_dict.items():
    options_etf.append({"label": val, "value": key })
    

def create_csi_chart(selected_values):

    fig = go.Figure()
    
    charts = []
    for idx, col in enumerate(selected_values):
        scat =  go.Scatter(x=csi_df.index, y=csi_df[col], name=col.upper(),
                        line = dict(color=color_scales[idx]),
                        hovertemplate="Ticker: %s<br>"%(col.upper())+\
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

    return fig



def create_etfs_chart(selected_values, lookback):
    quadrant_df_sub = quadrant_df.iloc[-lookback:]
    fig = go.Figure()

    xs, ys = [], []
    annotations = []
    
    for idx, ticker in enumerate(selected_values):
        col1, col2 = ticker+'_value1', ticker+'_gmean'
        x = quadrant_df_sub[col1].values.tolist()
        y = quadrant_df_sub[col2].values.tolist()
        xs.extend(x)
        ys.extend(y)

        fig.add_trace(go.Scatter(x=x,y=y, 
                                 customdata=[(ticker, ticker_dict[ticker])]*len(x),
                                 hovertemplate="Ticker: %{customdata[0]}<br>"+\
                                               "Name: %{customdata[1]}<br>"+\
                                               "X: %{x: .2f}<br>Y: %{y: .2f}",
                                 name=ticker_dict[ticker],
                                 mode='lines+markers',
                                 marker=dict(size=8, color=color_scales[idx]),
                                 line=dict(color=color_scales[idx])))

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
    fig.add_annotation(x=max(xs)+1.5, y=min(ys)-1.5, text="Reflation", showarrow=False, font_size=15, opacity=0.7)
    
    fig["layout"]["annotations"] += tuple(annotations)

    fig.update_layout(title="Historical Best ETFs",
                      xaxis_title="Strength vs S& 500",
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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__,  suppress_callback_exceptions=True,
                external_stylesheets=external_stylesheets,
                meta_tags=[{"name": "viewport", "content": "width=device-width"}])


csi_graph = dcc.Graph(figure=create_csi_chart(csi_df.columns[:2]), id="csi_chart", className="figure", style={"height":"70vh",})

etf_graph = dcc.Graph(figure=create_etfs_chart(list(ticker_dict.keys()), 5), id="etf_chart", className="figure", style={"height":"65vh",})


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
                style={"background-color": "#21252C", "padding":"2px","margin":"2px 0px 3px 0px", "select-value-color":"red"}
            )

etf_label = html.Label(children="ETFs MultiSelect", style={"color":"white", "margin":"3px"})      
            
multi_select_etf = dcc.Dropdown(
                id="multi_select_etf",
                options=options_etf,
                value=[opt["value"] for opt in options_etf],
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


left_pan = html.Div(children=[header2, radio_div], className="three columns", style={'backgroundColor': colors['background']})


main_dashboard = children=html.Div(children=[left_pan, right_pan], className="row")

app.layout = html.Div(style={'backgroundColor': colors['background'], "height":"120vh"}, 
                        children=[header, main_dashboard], className="row") # html.Hr(style={"color":"black", "padding":"0px", "margin":"0px"})


@app.callback(Output('right-pane', 'children'), [Input('chart-selection', 'value')])
def update_figure(selected_chart):
    if selected_chart == "CSI":
        multi_select_csi.value = csi_selects
        csi_graph.figure = create_csi_chart(multi_select_csi.value).to_dict()
        
        return [csi_label, multi_select_csi, csi_graph]
    else:
        etf_lookback_slider.value = slider_value
        multi_select_etf.value = etf_selects
        
        etf_graph.figure = create_etfs_chart(multi_select_etf.value, etf_lookback_slider.value).to_dict()
        
        return [slider_label, etf_lookback_slider, etf_graph, etf_label, multi_select_etf]
        
        
@app.callback(Output('csi_chart', 'figure'), [Input('multi_select_csi', 'value')])
def update_csi_lines_chart(selected_values):
    global csi_selects
    
    if not selected_values:
        selected_values = csi_df.columns[:2]
        multi_select_csi.value = selected_values
        
    csi_selects = selected_values
    
    return create_csi_chart(selected_values)
    

@app.callback(Output('etf_chart', 'figure'), [Input('multi_select_etf', 'value'), Input('lookback_slider', 'value')])
def update_etf_chart(selected_values, lookback_value):
    global slider_value, etf_selects
    
    if not selected_values:
        selected_values = list(ticker_dict.keys())
        multi_select_etf.value = selected_values
    
    slider_value = lookback_value
    etf_selects = selected_values
    
    return create_etfs_chart(selected_values, lookback_value)


server = app.server

if __name__ == '__main__':
    app.run_server(debug=True)
