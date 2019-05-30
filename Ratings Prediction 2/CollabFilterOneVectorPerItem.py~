# Make sure you use the autograd version of numpy (which we named 'ag_np')
# to do all the loss calculations, since automatic gradients are needed
import autograd.numpy as ag_np
import numpy as np
# Use helper packages
from AbstractBaseCollabFilterSGD import AbstractBaseCollabFilterSGD
from utils import load_dataset
import json
import os
import matplotlib.pyplot as plt
# Some packages you might need (uncomment as necessary)
import pandas as pd
## import matplotlib

# No other imports specific to ML (e.g. scikit) needed!


class CollabFilterOneVectorPerItem(AbstractBaseCollabFilterSGD):
    ''' One-vector-per-user, one-vector-per-item recommendation model.

    Assumes each user, each item has learned vector of size `n_factors`.

    Attributes required in param_dict
    ---------------------------------
    mu : 1D array of size (1,)
    b_per_user : 1D array, size n_users
    c_per_item : 1D array, size n_items
    U : 2D array, size n_users x n_factors
    V : 2D array, size n_items x n_factors

    Notes
    -----
    Inherits *__init__** constructor from AbstractBaseCollabFilterSGD.
    Inherits *fit* method from AbstractBaseCollabFilterSGD.
    '''

    def init_parameter_dict(self, n_users, n_items, train_tuple):
        ''' Initialize parameter dictionary attribute for this instance.

        Post Condition
        --------------
        Updates the following attributes of this instance:
        * param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values
        '''
        self.n_users = n_users
        self.n_items = n_items
        random_state = self.random_state # inherited
        self.param_dict = dict(
            mu=ag_np.ones(1),
            b_per_user=ag_np.ones(n_users),
            c_per_item=ag_np.ones(n_items),
            U=random_state.randn(n_users, self.n_factors),
            V=random_state.randn(n_items, self.n_factors),
            #n_users = self.n_users,
            #n_items = self.n_items,
            #random_state = self.random_state
            )

    def predict(self, user_id_N, item_id_N,
                mu=None, b_per_user=None, c_per_item=None, U=None, V=None):
        ''' Predict ratings at specific user_id, item_id pairs

        Args
        ----
        user_id_N : 1D array, size n_examples
            Specific user_id values to use to make predictions
        item_id_N : 1D array, size n_examples
            Specific item_id values to use to make predictions
            Each entry is paired with the corresponding entry of user_id_N

        Returns
        -------
        yhat_N : 1D array, size n_examples
            Scalar predicted ratings, one per provided example.
            Entry n is for the n-th pair of user_id, item_id values provided.
        '''
        N = user_id_N.size
        yhat_N = ag_np.ones(N)
        ### Handling of cold start for non existing users
        new_user_ids = []
        for user_id in user_id_N:
            if user_id > self.n_users:
                new_user_ids.append(user_id)
        
        if new_user_ids:
            b_new = np.zeros(max(new_user_ids) - self.n_users+1) + self.param_dict['b_per_user'].mean()
            u_new = np.zeros((max(new_user_ids) - self.n_users+1, self.n_factors)) + U.mean(axis=0)
            self.param_dict['U'] = np.vstack((self.param_dict['U'], u_new))
            self.param_dict['b_per_user'] = np.hstack((self.param_dict['b_per_user'], b_new))
            self.n_users = max(new_user_ids)+1
            self.param_dict['n_users'] = max(new_user_ids)+1
        
        ### Handling of cold start for non existing items
        new_item_ids = []
        for item_id in item_id_N:
            if item_id > self.n_items:
                new_item_ids.append(item_id)
        
        if new_item_ids:
            c_new = np.zeros(max(new_item_ids) - self.n_items+1) + self.param_dict['c_per_item'].mean()
            v_new = np.zeros((max(new_item_ids) - self.n_items+1, self.n_factors)) + V.mean(axis=0)
            self.param_dict['V'] = np.vstack((self.param_dict['V'], v_new))
            self.param_dict['c_per_item'] = np.hstack((self.param_dict['c_per_item'], c_new))
            self.n_items = max(new_item_ids)+1
            self.param_dict['n_items'] = max(new_item_ids)+1
            
            
        utv = (U[user_id_N] * V[item_id_N]).sum(axis=1)
        yhat_N = (mu + b_per_user[user_id_N] + c_per_item[item_id_N] + (utv)) * yhat_N
        return yhat_N


    def calc_loss_wrt_parameter_dict(self, param_dict, data_tuple):
        ''' Compute loss at given parameters

        Args
        ----
        param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values

        Returns
        -------
        loss : float scalar
        '''
        y_N = data_tuple[2]
        yhat_N = self.predict(data_tuple[0], data_tuple[1], **param_dict)
        loss_total = ag_np.power((y_N - yhat_N ), 2).sum() + (self.alpha * (ag_np.power(param_dict['U'],2).sum() + ag_np.power(param_dict['V'],2).sum()))
        return loss_total    

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
        
def save_train_valid_mae_loss_plot(model, path):
    plt.figure(figsize=(12,12))
    plt.plot(model.trace_epoch, model.trace_mse_train, c='red',label='Train Loss')
    plt.plot(model.trace_epoch, model.trace_mse_valid, c='green',label='Valid Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend(loc='best')
    plt.savefig(path)
    
    
if __name__ == '__main__':
    train_tuple, valid_tuple, test_tuple, n_users, n_items = load_dataset()
    min_train_mse, min_valid_mse, best_model, best_n_factors, best_alpha = 10.0, 10.0, 1, 2, 0.00
    i = 1
    for n_factors in [2, 10, 50]:
        for alpha in [0.00, 0.01, 0.1,]:
            print('Model Parameters : n_factors : %d, alpha : %f'%(n_factors,alpha))
            model = CollabFilterOneVectorPerItem(
                n_factors=n_factors, alpha=alpha,
                n_epochs=50, step_size=0.5)
            model.init_parameter_dict(n_users, n_items, train_tuple)
            model.fit(train_tuple, valid_tuple)
            
            if min(model.trace_mse_train) < min_train_mse and min(model.trace_mse_valid) < min_valid_mse:
                min_train_mse, min_valid_mse = min(model.trace_mse_train), min(model.trace_mse_valid)
                best_model,best_n_factors, best_alpha = i, n_factors, alpha
            
                
            save_train_valid_mae_loss_plot(model, os.path.join('plots','M3_%d.png'%i))
            
            submission_predictions = model.predict(test_tuple[0],test_tuple[1], **model.param_dict)
            
            #np.savetxt(os.path.join('test_preds',"M3_%d.txt"%i), submission_predictions,newline='\r\n')
            preds_df = pd.DataFrame({'user_ids':test_tuple[0],'item_ids':test_tuple[1], 'predicted_rating':submission_predictions})
            preds_df.to_csv(os.path.join('test_preds',"M3_%d.csv"%i),index=False)
    
            with open(os.path.join('model_params', 'M3_%d.json'%i),'w', encoding='utf-8') as f:
                json.dump(model.param_dict, f, cls=NumpyEncoder, indent=4)
            
            i = i + 1
    print('Best Train MSE : ', min_train_mse)
    print('Best Validation MSE : ', min_valid_mse)
    print('Best Model : ', best_model)
    print('Best Params : n_factors - %d, alpha - %f'%(best_n_factors,best_alpha))
