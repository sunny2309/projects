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

class CollabFilterOneScalarPerItem(AbstractBaseCollabFilterSGD):
    ''' One-scalar-per-user, one-scalar-per-item recommendation model.

    Attributes required in param_dict
    ---------------------------------
    mu : 1D array of size (1,)
    b_per_user : 1D array, size n_users
    c_per_item : 1D array, size n_items

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
        # TODO fix the lines below to have right dimensionality & values
        random_state = self.random_state # inherited
        self.param_dict = dict(
            mu=ag_np.zeros(1),
            b_per_user=ag_np.zeros(n_users),
            c_per_item=ag_np.zeros(n_items),
            )

    def predict(self, user_id_N, item_id_N,
                mu=None, b_per_user=None, c_per_item=None):
        ''' Predict ratings at specific user_id, item_id pairs

        Args
        ----
        user_id_N : 1D array, size n_examples
            Specific user_id values to use to make predictions
        item_id_N : 1D array, size n_examples
            Specific item_id values to use to make predictions
            Each entry is paired with the corresponding entry of user_id_N
        **param_dict : unpacked parameter dict

        Returns
        -------
        yhat_N : 1D array, size n_examples
            Scalar predicted ratings, one per provided example.
            Entry n is for the n-th pair of user_id, item_id values provided.
        '''
        # TODO: Update with actual prediction logic
        #print(user_id_N)
        N = user_id_N.size
        yhat_N = ag_np.ones(N)
        #print(N)
        #self.param_dict['mu'] = mu
        #self.param_dict['b_per_user'] = b_per_user
        #self.param_dict['c_per_item'] = c_per_item
        #print(mu.shape,b_per_user.shape,c_per_item.shape)
        yhat_N = (mu + b_per_user[user_id_N] + c_per_item[item_id_N]) * yhat_N
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
        # TODO compute loss
        # TIP: use self.alpha to access regularization strength
        y_N = data_tuple[2]
        yhat_N = self.predict(data_tuple[0], data_tuple[1], **param_dict)
        loss_total = ag_np.power((y_N - yhat_N ), 2).sum()
        return loss_total
    
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
        
def save_train_valid_mae_loss_plot(model, path):
    plt.figure(figsize=(12,12))
    plt.plot(model.trace_epoch, model.trace_mae_train, c='red',label='Train Loss')
    plt.plot(model.trace_epoch, model.trace_mae_valid, c='green',label='Valid Loss')
    #plt.plot(model.trace_epoch, model.trace_loss, c='red',title='TRACE LOSS')
    #plt.plot(model.trace_epoch, model.trace_loss, c='red',title='TRACE LOSS')
    plt.xlabel('Epochs')
    plt.ylabel('MAE Loss')
    plt.legend(loc='best')
    plt.savefig(path)
        
if __name__ == '__main__':
    train_tuple, valid_tuple, test_tuple, n_users, n_items = load_dataset()
    model = CollabFilterOneScalarPerItem(
        n_epochs=50, step_size=0.5)
    model.init_parameter_dict(n_users, n_items, train_tuple)
    model.fit(train_tuple, valid_tuple)
    
    print('Train MAE : ',min(model.trace_mae_train))
    print('Validation MAE : ',min(model.trace_mae_valid))
    
    save_train_valid_mae_loss_plot(model, os.path.join('plots','M2.png'))
    
    submission_predictions = model.predict(test_tuple[0],test_tuple[1], **model.param_dict)
    np.savetxt(os.path.join('test_preds', "M2.txt"), submission_predictions,newline='\r\n')
    
    with open(os.path.join('model_params', 'M2.json'),'w', encoding='utf-8') as f:
        json.dump(model.param_dict, f, cls=NumpyEncoder, indent=4)
        
    select_movies = pd.read_csv('data_movie_lens_100k/select_movies.csv')
    select_movies['c_parameter'] = model.param_dict['c_per_item'][select_movies.orig_item_id.values]
    select_movies.to_csv(os.path.join('select_movies','M2_select_movies_c_param.csv'), index=False)
