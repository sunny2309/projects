from bs4 import BeautifulSoup
import urllib
import pandas as pd

def parse_data_and_create_dataframe(rows, headers):
    table_data = []
    for row in rows[1:]:
        row_data = []
        spans = [values.find_all('span') for values in row.contents][1:] ## Finding all span values which has data for each cell of row.
        for span in spans:
            strings = []
            #print(span)
            for s in span:    
                if s.contents and isinstance(s.contents[0],str): ## Taking only cells which has data.
                    strings.append(s.contents[0])
                elif  s.get('class') and s.get('class')[0] == 'slash': ## Special slash condition for High/Low case. It's exception case.
                    strings.append('/')
            row_data.append(' '.join(strings)) ## Joining data of cell and appending it to row data.
        table_data.append(row_data) ## Appeding each row of table to list which will be converted to dataframe.

    past_10_days_data = pd.DataFrame(table_data, columns=headers)
    return past_10_days_data
    
def get_data_table(tables):
    for table in tables: 
        if table.get('class') and table.get('class')[0] == 'twc-table': ## Selecting table with class 'twc-table' which has weather data.
            return table
            
if __name__ == '__main__':
    ## Below is URL which we'll hit to get last 10 days data.
    res = urllib.request.urlopen('https://weather.com/en-IN/weather/tenday/l/ce0d3e9fc257a8eac296dc9f18bb35646c673240b8c68d03f15e5a618dbb0fc5')
    soup = BeautifulSoup(res.read(), 'html.parser') ## Initializing Beautiful Soup parser.
    tables = soup.findAll('table') ## Retrieving all table components
    
    table = get_data_table(tables) ## Getting table which has last 10 days data.
    headers = table.find_all('th') ## Parsing table to get headers of table.
    headers = [header.contents[0] for header in headers] ## Setting header contents
    
    print(len(headers), headers)
    
    rows = table.find_all('tr') ## Scraping all rows from table.
    
    past_10_days_data = parse_data_and_create_dataframe(rows, headers) ## Converting all rows of table to Pandas data frame.
    past_10_days_data.to_csv('weather.csv', index=False) ## Saving Pandas dataframe in same directory as CSV file.
