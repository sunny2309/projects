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
        # TODO fix the lines below to have right dimensionality & values
        # TIP: use self.n_factors to access number of hidden dimensions
        random_state = self.random_state # inherited
        self.param_dict = dict(
            mu=ag_np.ones(1),
            b_per_user=ag_np.ones(n_users),
            c_per_item=ag_np.ones(n_items),
            U=random_state.randn(n_users, self.n_factors),
            V=random_state.randn(n_items, self.n_factors),
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
        # TODO: Update with actual prediction logic
        N = user_id_N.size
        yhat_N = ag_np.ones(N)
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
        # TODO compute loss
        # TIP: use self.alpha to access regularization strength
        y_N = data_tuple[2]
        yhat_N = self.predict(data_tuple[0], data_tuple[1], **param_dict)
        #print(param_dict.keys())
        loss_total = ag_np.power((y_N - yhat_N ), 2).sum() + (self.alpha * (ag_np.power(param_dict['U'],2).sum() + ag_np.power(param_dict['V'],2).sum()))
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
    
    
def save_select_movies_embeddings_graph(V, titles , path, i):
    #print(V.shape)
    plt.figure(figsize=(12,12))
    plt.scatter(V[:,0],V[:,1], c=V.sum(axis=1), s=100, alpha=0.9, cmap=plt.cm.hsv)
    for i, title in enumerate(titles):
        plt.annotate(title, (V[i,0], V[i,1]))
        #plt.text(V[i,0]+0.03, V[i,1]+0.03, title, fontsize=9)
    plt.xlabel('V1')
    plt.ylabel('V2')
    plt.savefig(path)
    
if __name__ == '__main__':
    train_tuple, valid_tuple, test_tuple, n_users, n_items = load_dataset()
    min_train_mae, min_valid_mae, best_model, best_n_factors, best_alpha = 10.0, 10.0, 1, 2, 0.00
    i = 1
    for n_factors in [2, 10, 50]:
        for alpha in [0.00, 0.001, 0.01, 0.1,]:
            print('Model Parameters : n_factors : %d, alpha : %f'%(n_factors,alpha))
            model = CollabFilterOneVectorPerItem(
                n_factors=n_factors, alpha=alpha,
                n_epochs=50, step_size=0.5)
            model.init_parameter_dict(n_users, n_items, train_tuple)
            model.fit(train_tuple, valid_tuple)
            
            if min(model.trace_mae_train) < min_train_mae and min(model.trace_mae_valid) < min_valid_mae:
                min_train_mae, min_valid_mae = min(model.trace_mae_train), min(model.trace_mae_valid)
                best_model,best_n_factors, best_alpha = i, n_factors, alpha
            
                
            save_train_valid_mae_loss_plot(model, os.path.join('plots','M3_%d.png'%i))
            
            submission_predictions = model.predict(test_tuple[0],test_tuple[1], **model.param_dict)
            
            np.savetxt(os.path.join('test_preds',"M3_%d.txt"%i), submission_predictions,newline='\r\n')
            
            with open(os.path.join('model_params', 'M3_%d.json'%i),'w', encoding='utf-8') as f:
                json.dump(model.param_dict, f, cls=NumpyEncoder, indent=4)
            
            if n_factors == 2:
                select_movies = pd.read_csv('data_movie_lens_100k/select_movies.csv')
                V = model.param_dict['V'][select_movies.orig_item_id.values]  
                select_movies['V'] = [str(v) for v in V]
                select_movies.to_csv(os.path.join('select_movies','M3_select_movies_V_%d.csv'%i), index=False)          
                
                save_select_movies_embeddings_graph(V, select_movies['title'].values , os.path.join('select_movies','M3_%d.png'%i), i)
                
            i = i + 1
    print('Best Train MAE : ', min_train_mae)
    print('Best Validation MAE : ', min_valid_mae)
    print('Best Model : ', best_model)
    print('Best Params : n_factors - %d, alpha - %f'%(best_n_factors,best_alpha))
