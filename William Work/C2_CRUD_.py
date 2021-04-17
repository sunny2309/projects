import pandas as pd
import requests
import json
from tda import auth, client, orders

## Download Chrome Driver from this link :
## https://sites.google.com/a/chromium.org/chromedriver/downloads
token_path = '/home/sunny/Downloads/tmp/token.pickle'
api_key = 'IJJG7SMPLB12C2ZNRY2EWJXE9OVBAMOZ@AMER.OAUTHAP'
redirect_uri = 'https://localhost'

class C2_CRUD():

    def __init__(self, strategy = 'strategy1'):

        self.apikey = 'cGZ9TryF4sedNmYqZnh1YnCs0Hy7eJmaNCuJYNeSlvqIrtAvD8'
        self.strategy1 = '131798149' # CSI SP500
        self.strategy2 = '132159625' # WMG Sandbox
        self.strategy3 = '131997597' # CSI Stocks VS Bonds
        self.portfolio_configuration = []
        
        try:
            c = auth.client_from_token_file(token_path, api_key)
        except FileNotFoundError:
            print("Exception")
            from selenium import webdriver
            with webdriver.Chrome("/home/sunny/Downloads/Test API notebook/chromedriver") as driver:
                c = auth.client_from_login_flow(
                    driver, api_key, redirect_uri, token_path)
                    
        response = c.get_account(491477157, fields=c.Account.Fields.POSITIONS)
        account_details = response.json()
        
        for pos in account_details["securitiesAccount"]["positions"]:
            symbol, asset_type = pos["instrument"]["symbol"], pos["instrument"]["assetType"].lower()
            quantity = -pos["shortQuantity"] if pos["shortQuantity"] else pos["longQuantity"]
            self.portfolio_configuration.append([symbol, asset_type, quantity])
        
        #self.portfolio_configuration = [['SPY', 'stock', 30], ['MSFT', 'stock', 10], ['CAB1908R15', 'option', 2]] # [['MSFT', 'stock', 1], ['AMZN', 'stock', 1]]
        # Define which portfolio we are using
        if strategy == 'strategy1':
            self.strategy_queried = self.strategy1
        elif strategy == 'strategy2':
            self.strategy_queried = self.strategy2
        elif strategy == 'strategy3':
            self.strategy_queried = self.strategy3


    def get_full_c2_roster(self):

        url = 'https://api.collective2.com/world/apiv3/getSystemRoster?apikey={}'.format(self.apikey)
        payload = {}
        headers= {}

        return requests.request("POST", url, headers=headers, data = payload).json()


    def get_holdings(self):

        url = "https://api.collective2.com/world/apiv3/getDesiredPositions?apikey={}&systemid={}".format(self.apikey, self.strategy_queried)

        payload = {}
        headers= {}

        return requests.request("POST", url, headers=headers, data = payload).json()


    def create_positions_payload(self):

        data = {}
        repo = []

        # Loop through the input list of lists and add all positions to the repo
        for i in self.portfolio_configuration:
            repo.append({"symbol" : i[0], "typeofsymbol" : i[1], "quant" : i[2]})

        data["positions"] = repo

        # Convert to json format and return
        return json.dumps(data)


    def configure_portfolio(self):

        url = "https://api.collective2.com/world/apiv3/setDesiredPositions?apikey={}&systemid={}".format(self.apikey, self.strategy_queried)

        payload = self.create_positions_payload()
        headers = {
          'Content-Type': 'application/json'
        }

        return requests.request("POST", url, headers=headers, data = payload).json()


    def flatten_portfolio(self):

        self.portfolio_configuration = [["flat", "flat", 0]]

        return self.configure_portfolio()


c2_crud = C2_CRUD()
print("Account Positions : ")
print(c2_crud.portfolio_configuration)

######### Collective2 Options month codes
# month_code_calls = {
#     'January' : 'A',
#     'February' : 'B',
#     'March' : 'C',
#     'April' : 'D',
#     'May' : 'E',
#     'June' : 'F',
#     'July' : 'G',
#     'August' : 'H',
#     'September': 'I',
#     'October' : 'J',
#     'November' : 'K',
#     'December' : 'L'
# }
#
# month_code_puts = {
#     'January' : 'M',
#     'February' : 'N',
#     'March' : 'O',
#     'April' : 'P',
#     'May' : 'Q',
#     'June' : 'R',
#     'July' : 'S',
#     'August' : 'T',
#     'September' : 'U',
#     'October' : 'V',
#     'November' : 'W',
#     'December' : 'X'
# }
