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
## import pandas as pd
## import matplotlib

# No other imports specific to ML (e.g. scikit) needed!

class CollabFilterMeanOnly(AbstractBaseCollabFilterSGD):
    ''' Simple baseline recommendation model.

    Always predicts same scalar no matter what user/movie.

    Attributes required in param_dict
    ------------------------
    mu : 1D array of size (1,)

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
            Keys are string names of parameters (e.g. 'mu')
            Values are *numpy arrays* of parameter values
        '''
        # DONE for you. No need to edit.
        self.param_dict = dict(mu=ag_np.zeros(1))

    def predict(self, user_id_N, item_id_N, mu=None):
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
        yhat_N = mu * yhat_N
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
        loss_total = ag_np.power((y_N - yhat_N), 2).sum() + self.alpha * param_dict['mu']
        return loss_total

    
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
    model = CollabFilterMeanOnly(n_epochs=50)
    model.init_parameter_dict(n_users, n_items, train_tuple)
    model.fit(train_tuple, valid_tuple)
    print('Train MSE : ',min(model.trace_mse_train))
    print('Validation MSE : ',min(model.trace_mse_valid))
    print('Mean Rating Train : ', np.mean(train_tuple[2]))
    print('Mean Rating Validation : ', np.mean(valid_tuple[2]))
    save_train_valid_mae_loss_plot(model, os.path.join('plots','M1.png'))
    
    submission_predictions = model.predict(test_tuple[0],test_tuple[1], **model.param_dict)
    #np.savetxt(os.path.join('test_preds', "M1.txt"), submission_predictions,newline='\r\n')
    preds_df = pd.DataFrame({'user_ids':test_tuple[0],'item_ids':test_tuple[1], 'predicted_rating':predictions})
    preds_df.to_csv(os.path.join('test_preds', "M1.csv"),index=False)
    
