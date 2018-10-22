from user import User
import pandas as pd
from phe import paillier

X_COLUMNS = ['cylinders','displacement','horsepower','weight','acceleration','model year','origin']
Y_COLUMN = 'mpg'

if __name__ == '__main__':

    df = pd.read_csv('auto-mpg.csv')
    df = df[df.horsepower != '?']
    
    total_records = len(df)
    no_of_users = 2
    
    part_per_user = int(df.shape[0] / no_of_users)
    user_shares = []
    start = 0
    end = part_per_user
    for i in range(no_of_users):
        if i == no_of_users-1:
            end = df.shape[0]
            user_shares.append((start, end, df.iloc[start:,:]))
            continue
        user_shares.append((start,end, df.iloc[start:end, :]))
        start = end
        end += part_per_user
 
    pub, priv = paillier.generate_paillier_keypair(n_length = 1028)
    users = []
    for user in user_shares:
        users.append(User(user[2], X_COLUMNS,  Y_COLUMN,  total_records, user[0], user[1], pub))
        
