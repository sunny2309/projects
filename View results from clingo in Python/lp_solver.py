import pandas as pd
import re
from clyngor import ASP, solve
import warnings
import sys
import copy

warnings.filterwarnings('ignore')

answers = list(solve(sys.argv[1])) 
print('Number of Answers : %d'%len(answers))
print(answers[-1]) ## This one takes last answer from list of answers

## Below code loops through all entries and then seleccts only entries which has 'assign' word as first element
## to select pairs.
all_entries = []
for entry in answers[-1]:
    if entry[0] == 'assign':
        all_entries.append(entry)

print('Number of Entries : ', len(all_entries))
shift_mapping = {1:'morning', 2:'afternoon',3:'night', 4:'nightoff',5:'rest',6:'holiday'} ## Mapping from day number to day name 

## Below logic creates dictionary which has key as day number and value as another dictionary.
## Value ditionary holds key as day name like monring, evening etc and value of that dictionary will be list of employees in that
## shift
## Examples {1: {'morning': [2,3], 'evening':[5,6]}, 2: {'morning':[5,6], 'night': [7,8]}}

final_dict = {} 
for entry in all_entries:
    emp_id,shift_id, day_of_month = entry[1]
    emp_id,shift_id, day_of_month = int(emp_id), int(shift_id), int(day_of_month)
    if day_of_month not in final_dict:
        final_dict[day_of_month] = {'morning':[], 'afternoon':[], 'night':[], 'nightoff':[], 'rest':[], 'holiday':[]}
    final_dict[day_of_month][shift_mapping[shift_id]].append(emp_id)

df = pd.DataFrame(final_dict).loc[['morning','afternoon','night','nightoff','rest','holiday'],] ## Creating data frame from dict
df = df[sorted(df.columns)] ## Sorting based on columsn to get columns in order
df = df.astype('str') ## Converting all data to string to save in excel
for col in df.columns:
    df[col] = df[col].apply(lambda x: x.replace('[','').replace(']','')) ## Removing starting and end brackets from string values
df2 = copy.deepcopy(df) ## Creating 2nd copy of dataframe to save transpose of excel in case its too long for easy analysis
df2.columns = ['Day %d'%col for col in df2.columns] ## Adding 'Day' string before day numbers.
df2.to_excel('final_report.xlsx') 
df2.transpose().to_excel('final_report_trans.xlsx')
