import websocket
import talib

import krakenex
from pykrakenapi import KrakenAPI
from datetime import datetime

import time
import os

import json

import numpy as np
import pandas as pd
#from bs4 import BeautifulSoup


###### Data Feeds ----------------------------------------------------------------------------------------------------------

"""
I have proposed two format for us to use for our code structure. Which do you think is best? The goal will be to expand the functionality of the code so that it can be a market maker for multiple exchanges.
For example, we could use the code logic to trade crypto on the Kraken exchange, or, instead, trade stocks on either TD Ameritrade or Interactive Brokers.
"""

# Format proposal #1
class Exchange_Kraken:
    """
    Class for connecting datafeed to Kraken via websocket API
    """
    def __init__(self):
        self.api = krakenex.API(key="tnSOviP098GB+RnJrveKB/AAxruMvuSn76PW5wtvkrrp9Yy3ptF1XzFE",
                   secret="kfM624nMuyMOeNfbmNF7bfTgg1qwDwmLyn6L38jyywhFGsQsDnzifALo2OLIJSukJLpnpmMRcaNF5Bf8F4Zzog==")
        self.connection = KrakenAPI(self.api)
        
        self.STREAM_URL = "wss://ws.kraken.com"
        self.AUTH_URL = "wss://ws-auth.kraken.com"
    
    
    def subscribe_data(self, ohlc_size=1, currencies=["ETH/USD"]):
        # Function to subscribe to websocket API of the exchange
        
        subscription_msg = {
                          "event": "subscribe",
                          "pair": currencies,
                          "subscription": {
                            "interval": ohlc_size,
                            "name": "ohlc"
                          }
                        }
        
        ## Establish Connection
        self.data_conn = websocket.create_connection(self.STREAM_URL)
        ## Subscribe for 1 Min ETH data.
        self.data_conn.send(json.dumps(subscription_msg))    

    def trade(self, pair,typ,order_type,volume,validate=True):
        # Function to implement Trade
        '''
        pair: Crypto Currency
        typ: Order Type (buy or sell)
        order_type: market, limit, etc
        volume: trade quantity.
        validate: If True then order won't be actually placed but only parameters will be validated. If False then order will be placed.
        '''
        try:
            res = self.connection.add_standard_order(pair=pair, type=typ, ordertype=order_type, volume=volume, validate=validate)
            print("ORDER RESULT : {}".format(res))
        except Exception as e:
            print("Error : {}".format(e))
            return False
   
        return True

    def get_positions(self):
        # Function to get open current positions in the account
        return self.connection.get_open_positions()

    def get_open_orders(self):
        # Function to retrieve all open orders
        return self.connection.get_open_orders()
   
    def get_closed_orders(self):
        return self.connection.get_closed_orders()[0]
        
    def get_account_balance(self):
        return self.connection.get_account_balance()
        
        


class Exchange_IB:
    """
    Class for connecting datafeed to Interactive Brokers via websocket API
    """

    def subscribe_data(self, ohlc_size):
        # Function to subscribe to websocket API of the exchange
        pass

    def trade(self, ticker, qty, order_type):
        # Function to implement Trade
        pass

    def get_positions(self):
        # Function to get open current positions in the account
        pass

    def get_open_orders(self):
        # Function to retrieve all open orders
        pass


class Exchange_TDA:
    """
    Class for connecting datafeed to TD Ameritrade via websocket API
    Might be useful: https://tda-api.readthedocs.io/en/stable/streaming.html (although, there could be a better solution out there)
    """

    def subscribe_data(self, ohlc_size):
        # Function to subscribe to websocket API of the exchange
        pass

    def trade(self, ticker, qty, order_type):
        # Function to implement Trade
        pass

    def get_positions(self):
        # Function to get open current positions in the account
        pass

    def get_open_orders(self):
        # Function to retrieve all open orders
        pass


# Format proposal #2
class Exchange:

    def Kraken():
        pass

    def IB():
        pass


###### Market Maker --------------------------------------------------------------------------------------------------------


class MarketMaker:

    def __init__(self):
        self.default_bid = 1 # 1 would equal 1%, just like how Hummingbot formats it.
        self.default_ask = 1
        self.ohlc_size = 1 # 5 would mean 5 minute OHLCV bars
        self.data = []
        self.instrument = ["ETH/USD"]
        self.RSI_INTERVAL = 7 ## RSI with 7 days period
        self.RSI_OVERBOUGHT = 60
        self.RSI_OVERSOLD = 40
        self.TRADE_QUANTITY = 0.05
        self.in_position = False
        
        self.kraken = Exchange_Kraken()
        self.kraken.subscribe_data(ohlc_size=self.ohlc_size, currencies=self.instrument)
        self.parse_data()

    
    def parse_data(self):
        last_timestamp = None
        
        try:
            while True:
                msg = self.kraken.data_conn.recv()
                #print(msg)
                parsed_msg = json.loads(msg)
            
                if "event" not in parsed_msg:
                    if parsed_msg[1][1] != last_timestamp:
                        last_timestamp = parsed_msg[1][1]
                        row = parsed_msg[1]
                        row[1] = self.kraken.connection.unixtime_to_datetime(float(row[1]))
                        self.data.append(row)

                        ## Calling OnData when new entry of OHLC is received.
                        self.OnData()
                        
        except KeyboardInterrupt:
            print("Keyboard Interrupt")      
            
    
    def calculate_rsi(self):
        df =  pd.DataFrame(self.data,columns=["time", "DateTime", "Open","High","Low","Close","VolumeWeightedPrice","Volume","Count"])
        df["Close"] = df["Close"].astype("float")
        rsi = talib.RSI(df["Close"].values, self.RSI_INTERVAL)
        return rsi[-1], df["Close"].values[-1]

    def OnData(self):
        """
        The purpose of the OnData function is to mimic the code structure of QuantConnect. Meaning, everytime a new OHLCV bar is received from the data connection, what should our code do? (implement trading logic)
        """
        if len(self.data) > self.RSI_INTERVAL:
            last_rsi, last_close_price = self.calculate_rsi()
            print("Last RSI : {} and Close Price : ${:.2f} on {}".format(last_rsi, last_close_price, str(datetime.now())))

            if last_rsi > self.RSI_OVERBOUGHT:
                if self.in_position:
                    print("Sell Ethereum @ RSI : {} and Close Price : ${:.2f} on {}".format(last_rsi, last_close_price, str(datetime.now())))
                    order_status = self.kraken.trade("ETHUSD","sell","market",self.TRADE_QUANTITY)
                    #order_status = self.kraken.trade("ETHUSD","sell","market",self.TRADE_QUANTITY, False)
                    self.in_position = False
                else:
                    print("You don't have coin. Get it first to sell it.")

            if last_rsi < self.RSI_OVERSOLD:
                if self.in_position:
                    print("We already have coin. Nothing to Do.")
                else:
                    print("Buy Ethereum @ RSI : {} and Close Price : ${:.2f} on {}".format(last_rsi, last_close_price, str(datetime.now())))
                    order_status = self.kraken.trade("ETHUSD","buy","market",self.TRADE_QUANTITY)
                    #order_status = self.kraken.trade("ETHUSD","buy","market",self.TRADE_QUANTITY, False)
                    self.in_position = True
                    
                    

if __name__ == "__main__":
    mm = MarketMaker()
