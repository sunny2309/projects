### Kraken Imports
import krakenex
from pykrakenapi import KrakenAPI

#### Interactive Brokers Imports
import ibapi
from ibapi import wrapper
from ibapi.client import EClient
from ibapi.utils import iswrapper
from ibapi.contract import Contract
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.ticktype import TickTypeEnum
from ibapi.common import TagValueList
from ibapi.tag_value import TagValue


### Normal Imports
import time
import os

import json
import websocket
import talib

import numpy as np
import pandas as pd
import datetime
import warnings
import random
import math
import threading

warnings.filterwarnings("ignore")
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


    def subscribe_data(self, ohlc_size=1, currencies=[]):
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

    def trade(self, pair,typ,order_type,volume,validate=False, **kwargs):
        # Function to implement Trade
        '''
        pair: Crypto Currency
        typ: Order Type (buy or sell)
        order_type: market, limit, etc
        volume: trade quantity.
        validate: If True then order won't be actually placed but only parameters will be validated. If False then order will be placed.
        kwargs: This parameter is a dictionary which will have different parameters based on diff order types. E.g: price needs to be provided for limit orders.
        '''
        try:
            res = self.connection.add_standard_order(pair=pair, type=typ, ordertype=order_type, volume=volume, validate=validate, **kwargs)
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

    def get_open_positions(self):
        return self.connection.get_open_positions()


class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.hist_data = [] #Initialize variable to store candle
        self.acct_summary = []
        self.order_status = {}
        self.pending_orders = []
        self.positions = []
        self.trading_hours = {}
        self.tick_prices = {}
        
    def tickPrice(self, reqId, tickType, price, attrib):
        from ibapi.ticktype import TickTypeEnum
        #print("Tick ID : ", TickTypeEnum.to_str(int(tickType)), ", Price : ", price)
        self.tick_prices[TickTypeEnum.to_str(int(tickType))] = price
    
    def tickString(self, reqId, tickType, value):
        super().tickString(reqId, tickType, value)
        #print("TickString. TickerId:", reqId, "Type:", tickType, "Value:", value)
    
    def historicalData(self, reqId, bar):
        #print("Historical Data Req. Request ID : ", reqId)
        self.hist_data.append([bar.date, bar.open, bar.high, bar.low, bar.close, bar.average, bar.volume, bar.barCount])    
        
    def accountSummary(self, reqId, account, tag, value, currency):
        super().accountSummary(reqId, account, tag, value, currency)
        #print("Account Summary Data Req. Request ID : ", reqId)
        self.acct_summary.append([account, tag, value, currency])
        #print("AccountSummary. ReqId:", reqId, "Account:", account,"Tag: ", tag, "Value:", value, "Currency:", currency)
             
    def position(self, account, contract, position, avgCost):
        super().position(account, contract, position, avgCost)
        #print("Position.", "Account:", account, "Symbol:", contract.symbol, "SecType:",
        #      contract.secType, "Currency:", contract.currency,
        #      "Position:", position, "Avg cost:", avgCost)
        self.positions.append([account,contract.symbol,contract.secType,contract.currency,position,avgCost])
        
    def pnlSingle(self, reqId, pos, dailyPnL, unrealizedPnL, realizedPnL, value):
        super().pnlSingle(reqId, pos, dailyPnL, unrealizedPnL, realizedPnL, value)
        print("Daily PnL Single. ReqId:", reqId, "Position:", pos,
              "DailyPnL:", dailyPnL, "UnrealizedPnL:", unrealizedPnL,
              "RealizedPnL:", realizedPnL, "Value:", value)
        
    def nextValidId(self, orderId):
        super().nextValidId(orderId)
        self.nextorderId = orderId
        print('The next valid order id is: ', self.nextorderId)

    def orderStatus(self, orderId, status, filled, remaining, avgFullPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        self.order_status[orderId] = status
        print('orderStatus - orderid: ', orderId, 'status: ', status, 'filled ', filled, 'remaining ', remaining, 'lastFillPrice ', lastFillPrice)

    def openOrder(self, orderId, contract, order, orderState):
        print('openOrder id:', orderId, contract.symbol, contract.secType, '@', contract.exchange, ':', order.action, order.orderType, order.totalQuantity, orderState.status)

    def execDetails(self, reqId, contract, execution):
        print('Order Executed: ', reqId, contract.symbol, contract.secType, contract.currency, execution.execId, execution.orderId, execution.shares, execution.lastLiquidity)
        
    def openOrder(self, orderId, contract, order, orderState):
        super().openOrder(orderId, contract, order, orderState)
        #print("OpenOrder. PermId: ", order.permId, "ClientId:", order.clientId, " OrderId:", orderId, 
        #      "Account:", order.account, "Symbol:", contract.symbol, "SecType:", contract.secType,
        #      "Exchange:", contract.exchange, "Action:", order.action, "OrderType:", order.orderType,
        #      "TotalQty:", order.totalQuantity, "CashQty:", order.cashQty, 
        #      "LmtPrice:", order.lmtPrice, "AuxPrice:", order.auxPrice, "Status:", orderState.status)
        self.pending_orders.append([order.permId, order.clientId, orderId,order.account,contract.symbol,
                                    contract.secType,contract.exchange, order.action, order.orderType,
                                    order.totalQuantity, order.cashQty,order.lmtPrice,
                                    order.auxPrice, orderState.status])
        #order.contract = contract
        #self.permId2ord[order.permId] = order
    def openOrderEnd(self):
        super().openOrderEnd()
        print("OpenOrderEnd")
        #print("Received %d openOrders", len(self.permId2ord))
        
    def contractDetails(self, reqId, contractDetails):
        super().contractDetails(reqId, contractDetails)
        reg_session = contractDetails.liquidHours.split(";")[0].split("-")
        dt = reg_session[0].strip().split(":")[0]
        reg_session = " - ".join([d.split(":")[1] for d in reg_session])
        
        dt = datetime.datetime.strptime(dt, "%Y%m%d").strftime("%B %d - %Y")
        
        trading_session = contractDetails.tradingHours.split(";")[0].split("-")
        trading_session = " - ".join([d.split(":")[1] for d in trading_session])
        
        self.trading_hours["Date"] = dt
        self.trading_hours["Regular Trading Session"] = reg_session
        self.trading_hours["Total Available Hours"] = trading_session
        
        #print("Regular Trading Session : ", contractDetails.liquidHours.split(";")[0])
        
        #print("Total Available Hours   : ", contractDetails.tradingHours.split(";")[0])                
        
class Exchange_IB:
    """
    Class for connecting datafeed to Interactive Brokers via websocket API
    """
    
    def __init__(self):
        pass
    
    def createContract(self, symbol, sec_type="STK", exchange="SMART", currency="USD"):
        contract = Contract()
        contract.symbol = symbol
        contract.secType = sec_type
        contract.exchange = exchange
        contract.currency = currency

        return contract
    
    def createOrder(self, buy_sell="BUY", order_type="MKT", qty=1, duration="DAY", lmt_prc=""):
        order = Order()
        order.action = buy_sell
        order.orderType = order_type
        order.totalQuantity = qty
        order.Tif = duration
        order.outsideRth = True
        if order_type=="LMT" and lmt_prc:
            order.lmtPrice = lmt_prc

        return order 
        
    def start_app_in_background(self, app):
        #Start the socket in a thread
        api_thread = threading.Thread(target=app.run, daemon=True)
        api_thread.start()

        time.sleep(1) #Sleep interval to allow time for connection to server

    def retrieve_ohlc_data(self, ticker = "AAPL", start_date="", end_date='', bar_size="1 hour", measure="TRADES", sleep_time=5):
        if not start_date:
            print("Start Date Not Provided. Setting Start Date to Default 5 Days back from End Date.")
         
        try:
            dt2 = datetime.datetime.now() if not end_date else datetime.datetime.strptime(end_date, "%Y%m%d %H:%M:%S")
            dt2 = dt2 + datetime.timedelta(days=1)
            dt1 = datetime.datetime.strptime(start_date, "%Y%m%d %H:%M:%S") if start_date else dt2 - datetime.timedelta(days=5)
            
        except Exception as e:
            print(e)
        
        app = IBapi()
        app.connect('127.0.0.1', 7497, 123)

        self.start_app_in_background(app)

        ## We'll need to change sec_type and exchange, if we want to check other instruments than stocks.
        contract = self.createContract(symbol=ticker)
        
        app.reqHistoricalData(reqId=1,
                              contract=contract,
                              endDateTime=dt2.strftime("%Y%m%d %H:%M:%S"),
                              durationStr="1 D", # [S-Seconds,D-days,W-week,M-Month,Y-Year] default is S.
                              barSizeSetting=bar_size, # [1 sec,5 secs,15 secs,30 secs,1 min,2 mins,3 mins,5 mins,15 mins,20 mins,30 mins,1 hour,2 hours,3 hours,4 hours,8 hours,1 day, 1 week, 1 month]
                              whatToShow=measure, # [TRADES,MIDPOINT,BID,ASK,BID_ASK,ADJUSTED_LAST,HISTORICAL_VOLATILITY,OPTION_IMPLIED_VOLATILITY,REBATE_RATE,FEE_RATE,YIELD_BID,YIELD_ASK,YIELD_BID_ASK,YIELD_LAST]
                              useRTH=0,
                              formatDate=2,
                              keepUpToDate=False, chartOptions=[])

        time.sleep(sleep_time) #sleep to allow enough time for data to be returned

        df = pd.DataFrame(app.hist_data, columns=['DateTime', 'Open', 'High', 'Low','Close', 'Average', 'Volume', "BarCount"])
        
        ## Datetime format Handling.
        if bar_size in ['1 day', '1 week', '1 month', '1 W', '1 M']:  
            df['DateTime'] = pd.to_datetime(df['DateTime']) 
        else:
            df['DateTime'] = pd.to_datetime(df['DateTime'], unit='s') 
        
        df = df.set_index("DateTime")
        df = df[df.index >= "%d-%d-%d %d:%d:%d"%(dt1.year, dt1.month, dt1.day,dt1.hour,dt1.minute, dt1.second)]
        
        app.disconnect()

        return df
    
    def buy_order(self, ticker, order_type="MKT", qty=1, duration="DAY", lmt_price=""):
        try:
            app = IBapi()
            app.connect('127.0.0.1', 7497, 123)
            self.start_app_in_background(app)

            while True:
                if isinstance(app.nextorderId, int):
                    print('connected')
                    break
                else:
                    print('waiting for connection')
                    time.sleep(1)

            base_order = self.createOrder("BUY", order_type=order_type, qty=qty, duration=duration, lmt_prc=lmt_price)


            app.placeOrder(app.nextorderId, self.createContract(ticker), base_order)

            time.sleep(10)
                        
            app.disconnect()
            
        except Exception as e:
            print("Buy Order Failed : {}".format(e))
            return False
        
        return True

    def sell_order(self, ticker, order_type="MKT", qty=1, duration="DAY",  lmt_price=""):
        try:
            app = IBapi()
            app.connect('127.0.0.1', 7497, 123)
            self.start_app_in_background(app)

            while True:
                if isinstance(app.nextorderId, int):
                    print('connected')
                    break
                else:
                    print('waiting for connection')
                    time.sleep(1)

            base_order = self.createOrder("SELL", order_type=order_type, qty=qty, duration=duration, lmt_prc=lmt_price)


            app.placeOrder(app.nextorderId, self.createContract(ticker), base_order)

            time.sleep(10)
            
            app.disconnect()
            
        except Exception as e:
            print("Sell Order Failed : {}".format(e))
            return False
        
        return True
        
    def subscribe_data(self, ohlc_size):
        # Function to subscribe to websocket API of the exchange
        pass

    def trade(self, ticker="SPY", buy_sell="buy", order_type="MKT", qty=1, **kwargs):
        # Function to implement Trade
        if buy_sell == "buy":
            order_status = self.buy_order(ticker, order_type, qty, **kwargs)
        
        if buy_sell == "sell":
            order_status = self.sell_order(ticker, order_type, qty, **kwargs)
        
        return order_status
        
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
        self.reserve_price_strategy = 'avellaneda' # 'avellaneda', 'stoch_rsi'
        self.current_bias = None
        self.big_ask_spread_strategy = 'atr' # avellaneda, atr


        # Avellaneda/Stoikov market making settings
        self.current_reserve_price = None # Will get defined using the self.reserve_price() function.
        self.current_bid_ask_spread = None # Will get defined using the self.big_ask_spread() function.
        self.current_inventory_pct = 0 # This is a bounded number between -1 and 1 (where -1 means we are 100% short and 1 means we are 100% long)
        self.long_term_vol_interval = 250
        self.short_term_vol_interval = 14
        self.gamma = .2
        self.sigma = 1
        self.T = 1
        self.t = 0
        self.k = 1

        # Dollar balance information
        self.current_net_liquidation_value = None # The total dollar balance of the entire account (net liquidation value of everything)
        self.current_position_value = None # The total dollar balance of the traded asset (this can be either a positive or negative number depending on whether we are long or short the asset)
        self.current_position_qty = None # The total qty of the asset traded (e.g., .12 ETH )
        # We also need to define the maximum position balance of any long or short position (bounded by 1 or -1)
        self.max_long_position_balance = 1 # This means our max position size for long trades can be 100% of our dollar balance
        self.max_short_position_balance = -1 # This means our maximum short position size can be -100% of our dollar balance
        self.leverage = 2


        # Order refresh time (in seconds)
        self.order_refresh_time = 56 * 15
        self.next_order_datetime = datetime.datetime.now() # the current time must exceed this time for an action to take place





        self.kraken = Exchange_Kraken()
        self.kraken.subscribe_data(ohlc_size=self.ohlc_size, currencies=self.instrument)
        self.parse_data()


    def parse_data(self):
        last_timestamp = None

        # A few prerequisite steps we must take when starting our strategy
        # Step 1- We must load historical data from the exchange so the technical indicators will be useful immediately.
        self.warm_up_data()

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


    def cancel_outstanding_orders(self):
        """
        Function to cancel any outstanding orders for the traded asset.
        """
        open_orders = self.kraken.connection.get_open_orders()
        if len(open_orders) > 0:
            pair_name = self.instrument[0].replace("/", "")
            open_orders = open_orders[open_orders.descr_pair == pair_name]

            for transaction_id in open_orders.index:
                try:
                    cancelled_orders = self.kraken.connection.cancel_open_order(transaction_id)
                    print("ORDER : {} Cancelled. Result : {}".format(transaction_id, cancelled_orders))
                except Exception as e:
                    print("Transaction ID : {}, Error : {}".format(transaction_id, e))





    def calculate_rsi(self):
        df =  pd.DataFrame(self.data,columns=["time", "DateTime", "Open","High","Low","Close","VolumeWeightedPrice","Volume","Count"])
        df["Close"] = df["Close"].astype("float")
        rsi = talib.RSI(df["Close"].values, self.RSI_INTERVAL)
        return rsi[-1], df["Close"].values[-1]


    def warm_up_data(self):
        """
        Our technical indicators rely on historical data. When the script starts, no historical data exists because we must wait passively
        for data to arrive via websocket. This function should call a function that pulls in the most recent ~250 historical bars of OHLCV data for
        our traded asset.
        """
        # Insert code here for grabbing historical data
        hist_data = self.kraken.connection.get_ohlc_data(self.instrument, interval=self.ohlc_size, ascending=True)[0]
        self.data = hist_data.reset_index().values.tolist() # historical data for the ticker going 250+ bars


    def grab_current_inventory(self):
        """
        Function for retrieving the current inventory of the traded asset at the exchange.
        """


        account_balance = self.kraken.connection.get_account_balance()

        total_value = 0

        price_mapping = {}

        for pair, qty in zip(account_balance.index, account_balance.vol):
            ## Getting current prices of all currencies
            if pair == "ZUSD":
                total_value += qty
                #print("{} : {}".format(pair, qty))
                continue


            currency = "{}USD".format(pair[1:]) if len(pair) > 3 and pair.startswith("X") else "{}USD".format(pair)
            current_data = self.kraken.connection.get_ohlc_data(currency)[0]
            last_price = current_data["close"][0]
            total_value += (qty * last_price)

            #print("{} : {} : {}".format(pair, qty, last_price))

            price_mapping[pair] = last_price

            if len(pair) > 3 and pair.startswith("X"):
                if pair[1:] == self.instrument[0].split("/")[0]:
                    # The total dollar balance of the traded asset (this can be either a positive or negative number depending on whether we are long or short the asset)
                    self.current_position_value = (qty * last_price)

                    # The total qty of the asset traded (e.g., .12 ETH )
                    self.current_position_qty = qty

                    #print("{} : {} : {}".format(pair, qty, last_price))
            else:
                if pair == self.instrument[0].split("/")[0]:
                    # The total dollar balance of the traded asset (this can be either a positive or negative number depending on whether we are long or short the asset)
                    self.current_position_value = qty * last_price

                    # The total qty of the asset traded (e.g., .12 ETH )
                    self.current_position_qty = qty

                    #print("{} : {} : {}".format(pair, qty, last_price))

            # Pause for 1 second to not exceed api query limit
            time.sleep(1)

        for key, position in self.kraken.get_open_positions().items():
            pair = position["pair"].replace("ZUSD", "")

            if pair in price_mapping:
                last_price = price_mapping[pair]
            else:
                currency = "{}USD".format(pair[1:]) if len(pair) > 3 and pair.startswith("X") else "{}USD".format(pair)
                current_data = self.kraken.connection.get_ohlc_data(currency)[0]
                last_price = current_data["close"][0]

            # Turns out this is unnecessary - Apparently, the "full liquidation balance" is freely available by aggregating all the values in lines 307-315
            # total_value += (float(position["vol"]) * last_price)

            #print("{} : {} : {}".format(pair, position["vol"], last_price))

            if self.instrument[0].split("/")[0] in position["pair"]:
                if position["type"] == "buy":
                    self.current_position_value += (float(position["vol"]) * last_price)
                    self.current_position_qty += float(position["vol"])
                    #print("{} : {} : {}".format(pair, position["vol"], last_price))

                if position["type"] == "sell":
                    self.current_position_value -= (float(position["vol"]) * last_price)
                    self.current_position_qty -= float(position["vol"])
                    #print("{} : {} : {}".format(pair, position["vol"], last_price))


        # The total dollar balance of the entire account (net liquidation value of everything)
        self.current_net_liquidation_value = total_value

        # Calculate our current inventory (bounded value between -1 and 1)
        if self.current_position_value > 0:
            # Calculate the max dollar value that is allowed to be traded
            max_dollar_value_of_position = self.max_long_position_balance * self.current_net_liquidation_value
            self.current_inventory_pct = self.current_position_value / max_dollar_value_of_position

        elif self.current_position_value < 0:
            # Calculate the max dollar value that is allowed to be traded
            max_dollar_value_of_position = self.max_short_position_balance * self.current_net_liquidation_value
            self.current_inventory_pct = self.current_position_value / max_dollar_value_of_position * -1

        else:
            # Calculate the max dollar value that is allowed to be traded
            max_dollar_value_of_position = self.max_short_position_balance * self.current_net_liquidation_value
            self.current_inventory_pct = 0

        print("Current Net Liquidation Value        : $ {}".format(round(self.current_net_liquidation_value, 4)))
        print("Current Position Value               : $ {}".format(round(self.current_position_value, 4)))
        print("Current Position Quantity            : {}".format(round(self.current_position_qty, 4)))
        print("Current Inventory percentage         : {}".format(round(self.current_inventory_pct, 4)))




    def update_volatility(self):
        """
        A function to use for calculating volatility of the traded asset. We are trying to figure out if the traded asset is slightly more
        or slightly less volatile than its long term average. This way we can adjust the bid/ask spread to be wider or narrower depending on market conditions.
        """
        # Create dataframe out of self.data repo
        df =  pd.DataFrame(self.data,columns=["time", "DateTime", "Open","High","Low","Close","VolumeWeightedPrice","Volume","Count"])
        # Calc Average True Range for both long and short term bars
        long_term_vol = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod = self.long_term_vol_interval)
        short_term_vol = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod = self.short_term_vol_interval)
        # Calculate if the current volatility is lower or greater than long term volatility
        self.sigma = short_term_vol.iloc[-1] / long_term_vol.iloc[-1]
        # Save short term vol number for optional use in another strategy
        self.short_term_vol = short_term_vol.iloc[-1]

        print("Sigma                                : {}%".format(round(self.sigma, 4) * 100))

    def reserve_price(self):
        """
        Function that will calculate the reserve price for trading. The reserve price is the mean price we will use between the bid and ask spread.
        """

        # df =  pd.DataFrame(self.data,columns=["time", "DateTime", "Open","High","Low","Close","VolumeWeightedPrice","Volume","Count"])
        spread_data = self.kraken.connection.get_recent_spread_data(self.instrument)[0]
        mid_price = (spread_data.bid[0] + spread_data.ask[0])/2 # hypothetical code to find mid price of the traded asset (current bid price + current ask price divided by 2)

        if self.reserve_price_strategy == 'avellaneda':
            # Calculate current reserve price
            self.current_reserve_price = mid_price - (self.current_inventory_pct * self.gamma * (self.sigma ** 2) * (self.T - self.t))

        elif self.reserve_price_strategy == 'stoch_rsi':
            # Calc stoch_rsi
            df =  pd.DataFrame(self.data,columns=["time", "DateTime", "Open","High","Low","Close","VolumeWeightedPrice","Volume","Count"])
            k, d = talib.STOCHRSI(df['Close'])
            print(k.iloc[-1], d.iloc[-1])

            if k.iloc[-1] > d.iloc[-1]:
                print('bullish stoch_rsi bias')
                self.current_reserve_price = mid_price + (.2 * self.short_term_vol)
                self.current_bias = 'bullish'

            elif k.iloc[-1] < d.iloc[-1]:
                print('bearish stoch_rsi bias')
                self.current_reserve_price = mid_price - (.2 * self.short_term_vol)
                self.current_bias = 'bearish'

            elif k.iloc[-1] == d.iloc[-1]:
                if self.current_bias == 'bullish':
                    print('continuing bullish bias')
                    self.current_reserve_price = mid_price + (.2 * self.short_term_vol)
                    self.current_bias = 'bullish'

                elif self.current_bias == 'bearish':
                    print('continuing bearish bias')
                    self.current_reserve_price = mid_price - (.2 * self.short_term_vol)
                    self.current_bias = 'bearish'

                else:
                    print('No bias set... using mid price')
                    self.current_reserve_price = mid_price



        print("Current Reservation Price            : $ {}".format(round(self.current_reserve_price, 4)))
        print("Current Mid price                    : $ {}".format(round(mid_price, 4)))


    def bid_ask_spread(self):
        """
        The bid/ask spread is the distance between our simultaneous buy and sell orders.
        """
        if self.big_ask_spread_strategy == 'avellaneda':
            self.current_bid_ask_spread = (self.gamma * (self.sigma ** 2) * (self.T - self.t)) + ((2 / self.gamma) * np.log(1 + (self.gamma / self.k)))
        elif self.big_ask_spread_strategy == 'atr':
            self.current_bid_ask_spread = (self.short_term_vol * 1) # * self.sigma

        print("Current Bid Ask Spread               : {}".format(self.current_bid_ask_spread))


    def OnData(self):
        """
        The purpose of the OnData function is to mimic the code structure of QuantConnect. Meaning, everytime a new OHLCV bar is received from the data connection, what should our code do? (implement trading logic)
        """

        # Step 1 - check next_order_datetime to see if enough time passed to refresh our orders
        if datetime.datetime.now() < self.next_order_datetime:
            return

        """
        Our "pause" until the next_order_datetime has expired since we have made it to this step... now we must perform
        certain calculations before refreshing our buy and sell orders. Things we must calculate:

        1 - Volatility
        Volatility gets used when calculating the reserve price (explained below).

        2 - Reserve Price
        The reserve price will be close to the current price of the traded asset... but sometimes
        it will be a little higher or a little lower. The purpose of shifting our reference price is if we have a bullish or
        bearish bias, or, we need to flatten our inventory to get to a zero balance.

        3 - Bid & Ask spread
        This is the distance between our simultaneous buy and sell orders.
        """

        print("==================== Start ============================")
        print("Current Time                         : {}".format(datetime.datetime.now().strftime('%H:%M')))

        # Step 2 - Cancel any outstanding orders
        self.cancel_outstanding_orders()

        # Step 2 - We must retrieve the current portfolio balances before proceeding
        self.grab_current_inventory()

        # Step 3 - We should calculate the volatility of the traded asset
        self.update_volatility()

        # Step 4 - We must now determine our Reserve Price (i.e., a slightly shifted price based on current inventory of the asset being traded)
        self.reserve_price() # This function will calculate reserve price and save it as the self.current_reserve_price variable

        # Step 5 - What's our bid / ask spread ?
        self.bid_ask_spread()

        # Step 6 - Calculate trade sizes
        # We now must calculate our buy and sell order position sizes. We plan to be on both sides of the trade unless our inventory percentage is topped out in either direction.
        if self.current_inventory_pct > 0:
            # print('Inventory > 0', self.current_net_liquidation_value, self.max_short_position_balance, self.max_long_position_balance, self.current_position_value, self.current_inventory_pct)
            # Calculate the dollar value of the traded asset we will use for our bid order
            ask_size_dollar_amount = ((self.current_net_liquidation_value * self.max_short_position_balance) * -1)  # + (.5 * abs(self.current_position_value))
            bid_size_dollar_amount = (self.current_net_liquidation_value * self.max_long_position_balance) - self.current_position_value

        elif self.current_inventory_pct < 0:
            # print('Inventory < 0', self.current_net_liquidation_value, self.max_short_position_balance, self.max_long_position_balance, self.current_position_value, self.current_inventory_pct)
            # Calculate the dollar value of the traded asset we will use for our bid order
            ask_size_dollar_amount = (self.current_net_liquidation_value * self.max_short_position_balance * -1) - abs(self.current_position_value)
            bid_size_dollar_amount = (self.current_net_liquidation_value * self.max_long_position_balance) # + (.5*(self.current_position_value * -1))

        else:
            # print('Inventory == 0', self.current_net_liquidation_value, self.max_short_position_balance, self.max_long_position_balance, self.current_position_value, self.current_inventory_pct)
            ask_size_dollar_amount = (self.current_net_liquidation_value * self.max_short_position_balance)
            bid_size_dollar_amount = (self.current_net_liquidation_value * self.max_long_position_balance)


        # Calculate our bid and ask price
        ask_price = (self.current_reserve_price + (self.current_bid_ask_spread / 2))
        bid_price = (self.current_reserve_price - (self.current_bid_ask_spread / 2))
        # Calculate the quantity of the asset we want to submit trades
        sell_order_qty = ask_size_dollar_amount / ask_price #*.5
        buy_order_qty = bid_size_dollar_amount / bid_price #*.5

        # Submit new limit orders
        # This syntax needs to be cleaned up... I used made up language to fill in the methods below
        # validate parameter needs to be provided True for actually putting orders. Please check definition of trade() function above
        # price parameter needs to be provided for limit order if needed.
        print("Ask Price                            : $ {}".format(round(ask_price, 4)))
        print("Ask Dollar Amount                    : {}".format(round(ask_size_dollar_amount, 4)))
        print("Ask Quantity                         : {}".format(round(sell_order_qty, 4)))
        print("Bid Price                            : $ {}".format(round(bid_price, 4)))
        print("Bid Quantity                         : {}".format(round(buy_order_qty, 4)))
        print("Bid Dollar Amount                    : {}".format(round(bid_size_dollar_amount, 4)))

        print("==================== END ==============================\n")

        if ask_size_dollar_amount > 5:
            sell_order_status = self.kraken.trade(self.instrument[0].replace("/", ""), 'sell', 'limit', (1 * sell_order_qty), price="{:.2f}".format(ask_price), leverage = self.leverage) # Sell order
        else:
            sell_order_status = True

        if sell_order_status:
            if bid_size_dollar_amount > 5:
                buy_order_status = self.kraken.trade(self.instrument[0].replace("/", ""), 'buy', 'limit', (1 * buy_order_qty), price="{:.2f}".format(bid_price), leverage = self.leverage) # Buy order

        # finally, set the self.next_order_datetime to a forward date so that the bot rests until then
        self.next_order_datetime = datetime.datetime.now() + datetime.timedelta(seconds = self.order_refresh_time)
        print('Next update time: ', self.next_order_datetime)


class MarketMaker_IB:
    """
    Market maker logic designed to use Interactive Brokers connection.
    """

    def __init__(self):
        self.ohlc_size = 1 # 5 would mean 5 minute OHLCV bars
        self.data = None
        self.instrument = "AAPL"
        self.RSI_INTERVAL = 7 ## RSI with 7 days period
        self.RSI_OVERBOUGHT = 60
        self.RSI_OVERSOLD = 40
        self.TRADE_QUANTITY = 1
        self.in_position = False


        self.ib = Exchange_IB()
        self.parse_data()
    
    def warm_up_data(self):
        end_date = datetime.datetime.today().strftime("%Y%m%d %H:%M:00")
        start_date = (datetime.datetime.today() - datetime.timedelta(days=1)).strftime("%Y%m%d %H:%M:00")
        df =  self.ib.retrieve_ohlc_data(ticker=self.instrument, start_date=start_date, end_date=end_date, bar_size="1 min")
        
        self.data = df.copy()
        
        
    def parse_data(self):
        # A few prerequisite steps we must take when starting our strategy
        # Step 1- We must load historical data from the exchange so the technical indicators will be useful immediately.
        self.warm_up_data()
        
        time.sleep(60)
        
        try:
            while True:
                last_dt = self.data.index[-1]
                last_date_fmt = "%d%02d%02d %d:%d:00"%(last_dt.year, last_dt.month, last_dt.day, last_dt.hour, last_dt.minute)
                end_date = datetime.datetime.today().strftime("%Y%m%d %H:%M:00")    
                
                print(last_date_fmt, end_date)
                
                latest_data = self.ib.retrieve_ohlc_data(ticker=self.instrument, start_date=last_date_fmt, end_date=end_date, bar_size="1 min")
                
                if len(latest_data) > 1:
                    self.data = pd.concat(self.data, latest_data[1:])
                    ## Calling OnData when new entry of OHLC is received.
                    self.OnData()
                    
                time.sleep(60)

        except KeyboardInterrupt:
            print("Keyboard Interrupt")


    def calculate_rsi(self):
        rsi = talib.RSI(self.data["Close"].values, self.RSI_INTERVAL)
        return rsi[-1], self.data["Close"].values[-1]


    def OnData(self):
        if len(self.data) > self.RSI_INTERVAL:
            last_rsi, last_close_price = self.calculate_rsi()
            print("Last RSI : {} and Close Price : ${:.2f} on {}".format(last_rsi, last_close_price, str(datetime.datetime.now())))

            if last_rsi > self.RSI_OVERBOUGHT:
                if self.in_position:
                    print("Sell {} @ RSI : {} and Close Price : ${:.2f} on {}".format(self.instrument, last_rsi, last_close_price, str(datetime.datetime.now())))
                    order_status = self.ib.trade(ticker=self.instrument,buy_sell="sell",order_type="MKT",qty=self.TRADE_QUANTITY)
                    #order_status = self.ib.trade(ticker=self.instrument,buy_sell="sell",order_type="LMT",qty=self.TRADE_QUANTITY, lmt_price="123.456")
                    
                    self.in_position = False
                else:
                    print("You don't have stocks. Get it first to sell it.")

            if last_rsi < self.RSI_OVERSOLD:
                if self.in_position:
                    print("We already have stocks. Nothing to Do.")
                else:
                    print("Buy {} @ RSI : {} and Close Price : ${:.2f} on {}".format(self.instrument, last_rsi, last_close_price, str(datetime.datetime.now())))
                    order_status = self.ib.trade(ticker=self.instrument,buy_sell="buy",order_type="MKT",qty=self.TRADE_QUANTITY)
                    #order_status = self.ib.trade(ticker=self.instrument,buy_sell="buy",order_type="LMT",qty=self.TRADE_QUANTITY, lmt_price="123.456")
                    
                    self.in_position = True




if __name__ == "__main__":
    mm = MarketMaker_IB()
