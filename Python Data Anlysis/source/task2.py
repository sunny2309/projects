import numpy as np

data = np.loadtxt('kickstarter.csv', delimiter= ',' , dtype='str') # load the data file 

def tidy_string(str): #  this function tidy the string
    return str.replace("'", "").replace('b', '')  # replace the dummy characters

info = [] 
category_list = []
for i in range(1, len(data)):  # for all data item
    tmp_info = {}
    tmp_info['state'] = tidy_string(data[i][4]) # get the state of one row
    tmp_info['staff_pick'] = tidy_string(data[i][8]) # get the staff picker of one row
    tmp_info['category'] = tidy_string(data[i][12]) # get the category of one row
    tmp_info['spotlight'] = tidy_string(data[i][13]) # get the sport light of one row
    category_list.append(tmp_info['category']) # append the category to category list
    info.append(tmp_info)  # append one category and state pair to info list

categories = list(set(category_list)) # get the unique categories

staff_total_cnt = 0
staff_success_cnt = 0

spot_total_cnt = 0
spot_success_cnt = 0

for category in categories: # for all type of categories
    one_category_state_staffpick = []
    one_category_state_spotlight = []
    
    for i in range(len(info)):
        if info[i]['category'] == category and info[i]['staff_pick'] == 'TRUE':   
            one_category_state_staffpick.append(info[i]['state'])  # we build the one category state list for selected a category for staff pick feature
        elif info[i]['category'] == category and info[i]['spotlight'] == 'TRUE':   
            one_category_state_spotlight.append(info[i]['state'])  # we build the one category state list for selected a category for spotlight feature
            
    percent_staff = 0
    if len(one_category_state_staffpick) != 0:    
        success_categories_staff = np.where(np.array(one_category_state_staffpick) == 'successful')  # we get the index list of categories which it's state is 'successful' for staff pick feature
        percent_staff = len(success_categories_staff[0]) / len(one_category_state_staffpick)  *100  # calculate the percentage of successful categories for staff pick feature
        
        staff_total_cnt = staff_total_cnt + len(one_category_state_staffpick)
        staff_success_cnt = staff_success_cnt + len(success_categories_staff)
        
    print("staff pick featured categories : {0}  success percentage : {1}% ".format(category, percent_staff) )  # print the percentage for staff pick feature
    
    percent_spotlight = 0
    if len(one_category_state_spotlight) != 0:
        success_categories_spotlight = np.where(np.array(one_category_state_spotlight) == 'successful')  # we get the index list of categories which it's state is 'successful' for spotlight feature
        percent_spotlight = len(success_categories_spotlight[0]) / len(one_category_state_spotlight)  *100    # calculate the percentage of successful categories for spotlight feature
        
        spot_total_cnt = spot_total_cnt + len(one_category_state_spotlight)
        spot_success_cnt = spot_success_cnt + len(success_categories_spotlight)        
    print("spotlight featured categories : {0}  success percentage : {1}% ".format(category, percent_spotlight) )  # print the percentage for spotlight feature
    
print("==========================================")
print("staff pick featured categories success percentage : {0}% ".format( staff_success_cnt / staff_total_cnt * 100) )  # print the percentage for staff pick feature
print("spotlight featured categories  success percentage : {0}% ".format( spot_success_cnt / spot_total_cnt * 100) )  # print the percentage for spotlight feature
    