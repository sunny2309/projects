import numpy as np
import pandas as pd


import plotly.express as px


import dash
import dash_core_components as dcc
import dash_html_components as html

from collections import Counter
import warnings

warnings.filterwarnings('ignore')
np.set_printoptions(precision=3)
pd.set_option('precision',3)


cols = ['Date of submission','PAC WellBore ID','PAC String Type','WELL STATUS','WELL STATUS L1','HEALTHY/UNHEALTHY','UNHEALTHY CATEGORY','ISSUE','WFAM Score','Age','Age Factors','Score','Shallow Gas','Score2','Metal Loss Above Top Packer','Score3','Cement','Score4','Casing Thread','Score5','Corrosive Fluid','Score6','Final Score','WIRE']
well_integrity = pd.read_csv('~/mysite/Data Well Integrity.csv')
well_integrity = well_integrity[well_integrity["WIRE"] != "#VALUE!"]
well_integrity = well_integrity[cols]

well_integrity['WELL STATUS'] =  ['Abandon' if val in ['Abandoned', 'Abandonment'] else 'Active' if val=='Active' else 'Idle' if val=='Idle' else val.capitalize() for val in well_integrity['WELL STATUS']]

well_status_dist = well_integrity.groupby(by=['WELL STATUS', 'WELL STATUS L1']).count()[['PAC String Type']]
well_status_dist = well_status_dist.rename(columns={'PAC String Type': 'Count'})
well_status_dist = well_status_dist.sort_values(by=['WELL STATUS', 'Count'], ascending=False)
well_status_dist = well_status_dist.reset_index()
well_status_dist = well_status_dist.sort_values(by="Count", ascending=False)

health_dist = well_integrity.groupby(by=['HEALTHY/UNHEALTHY', 'UNHEALTHY CATEGORY']).count()[['PAC String Type']]
health_dist = health_dist.rename(columns={'PAC String Type': 'Count'})
health_dist = health_dist.sort_values(by=['HEALTHY/UNHEALTHY', 'Count'], ascending=False)
health_dist = health_dist.reset_index()
health_dist = health_dist.sort_values(by="Count", ascending=False)

unhealthy_dist = well_integrity[well_integrity['HEALTHY/UNHEALTHY'] == 'Unhealthy']
unhealthy_dist = unhealthy_dist.groupby(by=['UNHEALTHY CATEGORY', 'ISSUE']).count()[['PAC String Type']]
unhealthy_dist = unhealthy_dist.rename(columns={'PAC String Type': 'Count'})
unhealthy_dist = unhealthy_dist.sort_values(by=['UNHEALTHY CATEGORY', 'Count'], ascending=False)
unhealthy_dist = unhealthy_dist.reset_index()
unhealthy_dist = unhealthy_dist.sort_values(by="Count", ascending=False)

wire_dist_by_well_status = well_integrity.groupby(by=['UNHEALTHY CATEGORY', 'WIRE']).count()[['PAC String Type']]
wire_dist_by_well_status = wire_dist_by_well_status.rename(columns={'PAC String Type': 'Count'})
wire_dist_by_well_status = wire_dist_by_well_status.sort_values(by=['UNHEALTHY CATEGORY', 'Count'], ascending=False)
wire_dist_by_well_status = wire_dist_by_well_status.reset_index()
wire_dist_by_well_status = wire_dist_by_well_status.sort_values(by="Count", ascending=False)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

header = html.H2(children="Well Integrity Risk Escalation (WIRE) Dashboard By Faiz Fablillah", style={"margin": "0px"})

chart1 = px.bar(health_dist,
                         x='HEALTHY/UNHEALTHY', y='Count',
                         color='UNHEALTHY CATEGORY',
                         height=580,
                         #width=600,
                         title='Dist. Of "UNHEALTHY CATEGORY" in "HEALTHY/UNHEALTHY"')

chart1.update_layout(legend_orientation="h", paper_bgcolor=colors['background'], plot_bgcolor=colors['background'], font={'color': colors['text']})

for trace in chart1.data:
    trace.name = trace.name.split('=')[1]

graph1 = dcc.Graph(
        id='graph1',
        #config={'displayModeBar': False},
        #className='five columns',
        figure=chart1
    )

chart2 = px.bar(well_status_dist,
                         x='WELL STATUS', y='Count',
                         color='WELL STATUS L1',
                         height=580,
                         #width=600,
                         title='Dist. Of "WELL STATUS L1" in "WELL STATUS"')

chart2.update_layout(legend_orientation="h", paper_bgcolor=colors['background'], plot_bgcolor=colors['background'], font={'color': colors['text']})

for trace in chart2.data:
    trace.name = trace.name.split('=')[1]

graph2 = dcc.Graph(
        id='graph2',
        #config={'displayModeBar': False},
        #className='five columns',
        figure=chart2
    )

chart3 = px.bar(wire_dist_by_well_status.reset_index(),
                         x='UNHEALTHY CATEGORY', y='Count',
                         height=580,
                         #width=600,
                         color='WIRE',
                         title='Dist. Of "WIRE" in "UNHEALTHY CATEGORY"')

chart3.update_layout(legend_orientation="h", paper_bgcolor=colors['background'], plot_bgcolor=colors['background'], font={'color': colors['text']} )

for trace in chart3.data:
    trace.name = trace.name.split('=')[1]

graph3 = dcc.Graph(
        id='graph3',
        #config={'displayModeBar': False},
        #className='five columns',
        figure=chart3
    )

chart4 = px.bar(unhealthy_dist.reset_index(),
                         x='UNHEALTHY CATEGORY', y='Count',
                         height=580,
                         #width=600,
                         color='ISSUE',
                         title='Dist. Of "ISSUE" in "UNHEALTHY CATEGORY"')

chart4.update_layout(legend_orientation="h", paper_bgcolor=colors['background'], plot_bgcolor=colors['background'], font={'color': colors['text']})

for trace in chart4.data:
    trace.name = trace.name.split('=')[1]

graph4 = dcc.Graph(
        id='graph4',
        #config={'displayModeBar': False},
        #className='five columns',
        figure=chart4
    )



healthy_unhealthy_cnt = Counter(well_integrity["HEALTHY/UNHEALTHY"])

table1 = html.Table(
            [
            html.Tr(
                [html.Td(" HEALTHY ", style={"background-color":"green", "text-align":"center", "color":"white", "border": "1px solid black","padding":"2px", "font-weight":"bold", "width": "50%"}),
                html.Td("UNHEALTHY", style={"background-color":"red", "text-align":"center","color":"white",  "border": "1px solid black", "padding":"2px", "font-weight":"bold", "width": "50%"})
                ]),
            html.Tr(
                [
                html.Td("%.1f%%"%(healthy_unhealthy_cnt["Healthy"]*100/well_integrity.shape[0]), style={"background-color":"green", "text-align":"center", "color":"white", "border": "1px solid black", "padding":"2px", "font-weight":"bold"}),
                html.Td("%.1f%%"%(healthy_unhealthy_cnt["Unhealthy"]*100/well_integrity.shape[0]), style={"background-color":"red", "text-align":"center","color":"white", "border": "1px solid black","padding":"2px",  "font-weight":"bold"})
                ]),
            html.Tr(
                [
                html.Td("%d"%healthy_unhealthy_cnt["Healthy"], style={"background-color":"green", "text-align":"center", "color":"white","border": "1px solid black","padding":"2px", "font-weight":"bold"}),
                html.Td("%d"%healthy_unhealthy_cnt["Unhealthy"], style={"background-color":"red", "text-align":"center","color":"white", "border": "1px solid black","padding":"2px", "font-weight":"bold"})
                ]),
            html.Tr(
                [
                html.Td("Strings", style={"background-color":"green", "text-align":"center", "color":"white", "border": "1px solid black","padding":"2px", "font-weight":"bold"}),
                html.Td("Strings", style={"background-color":"red", "text-align":"center","color":"white", "border": "1px solid black","padding":"2px", "font-weight":"bold"})
                ]),
           ], style={"justify-content":"center", "align-items": "center", "float": "center", "width":"100%"})

unhealthy_cat_dist = Counter(well_integrity["UNHEALTHY CATEGORY"])

table2 = html.Table(
            [
            html.Tr(
                [
                html.Td(children="P1- Serious", style={"background-color":"red", "text-align":"center","border": "1px solid black","padding":"2px", "font-weight":"bold", "width": "70%", "color":"black"}),
                html.Td(unhealthy_cat_dist["P1- Serious"], style={"text-align":"center","border": "1px solid black", "padding":"2px", "font-weight":"bold", "width": "30%"})
                ]),
            html.Tr(
                [
                html.Td("P2 - MOC", style={"background-color":"orange", "text-align":"center", "border": "1px solid black","padding":"2px", "font-weight":"bold", "color":"black"}),
                html.Td(unhealthy_cat_dist["P2 - MOC"], style={"text-align":"center","border": "1px solid black", "padding":"2px", "font-weight":"bold"})
                ]),
            html.Tr(
                [
                html.Td("P2 - Exceed Limit", style={"background-color":"orange", "text-align":"center", "border": "1px solid black","padding":"2px", "font-weight":"bold", "color":"black"}),
                html.Td(unhealthy_cat_dist["P2 - Exceed Limit"], style={"text-align":"center","border": "1px solid black", "padding":"2px", "font-weight":"bold"})
                ]),
            html.Tr(
                [html.Td("P3-Within Limit", style={"background-color":"yellow", "text-align":"center","border": "1px solid black", "padding":"2px", "font-weight":"bold", "color":"black"}),
                html.Td(unhealthy_cat_dist["P3-Within Limit"], style={"text-align":"center", "border": "1px solid black","padding":"2px", "font-weight":"bold"})
                ]),
            html.Tr(
                [html.Td("Intact", style={"background-color":"green", "text-align":"center","border": "1px solid black", "padding":"2px", "font-weight":"bold", "color":"black"}),
                html.Td(unhealthy_cat_dist["Intact"], style={"text-align":"center","border": "1px solid black", "padding":"2px", "font-weight":"bold"})
                ]),
           ], style={"justify-content":"center", "align-items": "center", "float": "center", "width": "100%",})


row1 = html.Div(children=[html.Div([html.Br(),
                                    html.Br(),
                                    html.Br(),
                                    html.Br(),
                                    html.Br(),
                                    #html.Img(src="static/RTI_Logo.jpeg", style={"width":"75x", "height":"75px"}),

                                    html.H5("MALAYSIA WELLS HEALTHINESS", style={"color":colors['text'], "font-weight":"bold"}),
                                    html.Hr(style={"margin":"0px"}),
                                    html.H5("TOTAL STRINGS", style={"font-weight":"bold"}),
                                    html.H4("%d"%well_integrity.shape[0], style={"font-weight":"bold", }),
                                    html.Div(table1, style={"justify-content":"center", "align-items": "center", 'backgroundColor': colors['background'],"padding":"11px"}),
                                    html.Br(),
                                    html.Br(),
                                    html.Br(),
                                    html.Br(),], className="two columns", style={'backgroundColor': colors['background'], "width":"16%",}),
                                    html.Div(graph1, className="five columns", style={'backgroundColor': colors['background'], "margin":"0px", "padding":"0px", "width":"42%"}),
                                    html.Div(graph2, className="five columns", style={'backgroundColor': colors['background'], "margin":"0px", "padding":"0px", "width":"42%"})], style={'backgroundColor': colors['background']})

row2 = html.Div(children=[html.Div([html.Br(),
                                    html.Br(),
                                    html.Br(),
                                    html.Br(),
                                    html.Br(),
                                    html.H5("UNHEALTHY STRINGS", style={"color":colors['text'], "font-weight":"bold"}),
                                    html.Hr(style={"margin":"0px", "padding":"2px"}),
                                    html.H5("CATEGORY & PRIORITY", style={"font-weight":"bold"}),
                                    html.H4("%d Strings"%np.sum([unhealthy_cat_dist["P1- Serious"],
                                                                 unhealthy_cat_dist["P2 - MOC"],
                                                                 unhealthy_cat_dist["P2 - Exceed Limit"]]), style={"color":colors['text'], "font-weight":"bold"}),
                                    html.Div(table2, style={"justify-content":"center", "align-items": "center", 'backgroundColor': colors['background'], "padding":"10px"}),
                                    html.Br(),
                                    html.Br(),
                                    html.Br(),
                                    html.Br(),], className="two columns", style={'backgroundColor': colors['background'], "width":"16%"}),
                                    html.Div(graph3, className = "five columns", style={'backgroundColor': colors['background'], "margin":"0px", "padding":"0px","width":"42%"}),
                                    html.Div(graph4, className = "five columns", style={'backgroundColor': colors['background'], "margin":"0px", "padding":"0px","width":"42%"}),])

layout = html.Div(children=[header, row1, row2], style={"text-align": "center", 'backgroundColor': colors['background'], 'color': colors['text'], })

app.layout = layout
server = app.server

if __name__ == "__main__":
    app.run_server(debug=True)



# This file contains the WSGI configuration required to serve up your
# web application at http://<your-username>.pythonanywhere.com/
# It works by setting the variable 'application' to a WSGI handler of some
# description.
#
# The below has been auto-generated for your Flask project

import sys

# add your project directory to the sys.path
project_home = u'/home/sunny2309/mysite'
if project_home not in sys.path:
    sys.path = [project_home] + sys.path

# import flask app but need to call it "application" for WSGI to work
from well_integrity_analysis import app as plotly_app

application = plotly_app.server


