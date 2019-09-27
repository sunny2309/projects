import numpy as np

data = np.loadtxt('kickstarter.csv', delimiter= ',' , dtype='str') # load the data file 

def tidy_string(str): #  this function tidy the string
    return str.replace("'", "").replace('b', '')  # replace the dummy characters

info = [] 
category_list = []
for i in range(1, len(data)):  # for all data item
    tmp_info = {}
    tmp_info['state'] = tidy_string(data[i][4]) # get the state of one row
    tmp_info['category'] = tidy_string(data[i][12]) # get the category of one row
    category_list.append(tmp_info['category']) # append the category to category list
    info.append(tmp_info)  # append one category and state pair to info list
    
categories = list(set(category_list)) # get the unique categories

max_category = ''
max_percent = 0
min_category = ''
min_percent = 100
for category in categories: # for all type of categories
    one_category_state = []
    for i in range(len(info)):
        if info[i]['category'] == category:   
            one_category_state.append(info[i]['state'])  # we build the one category state list for selected a category
    
    success_categories = np.where(np.array(one_category_state) == 'successful')  # we get the index list of categories which it's state is 'successful'
    percent = len(success_categories[0]) / len(one_category_state)  *100
    print("categories : {0}  success percentage : {1}% ".format(category, percent) )  # print the percentage
    if percent > max_percent:  # get the max percent and max category
        max_percent = percent
        max_category = category
    if percent < min_percent:  # get the min percent and min category
        min_percent = percent
        min_category = category    
        
print("-----------------------------------------------")
print("Max category : {0}  success percentage : {1}% ".format(max_category, max_percent) )  # print the percentage
print("Min category : {0}  success percentage : {1}% ".format(min_category, min_percent) )  # print the percentage