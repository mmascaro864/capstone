import numpy as np
import pandas as pd
import recommender_functions as rf

class Recommender():
    '''
    This Recommender uses FunkSVD to make predictions of exact ratings.  And uses either FunkSVD or a Knowledge Based recommendation (highest ranked) to make recommendations for users.  Finally, if given a movie, the recommender will provide movies that are most similar as a Content Based Recommender.
    '''
    def __init__(self):
        '''
        I didn't have any required attributes needed when creating my class.
        '''
        
    def fit(self, train_df, val_df, latent_features=4, learning_rate=0.005, iters=100):
        '''
        Purpose:
            This function performs matrix factorization using a basic form of FunkSVD with no regularization
    
        INPUT:
            ratings_mat - (numpy array) a matrix with users as rows, movies as columns, and ratings as values
            latent_features - (int) the number of latent features used
            learning_rate - (float) the learning rate 
            iters - (int) the number of iterations
    
        OUTPUT:
            user_mat - (numpy array) a user by latent feature matrix
            offer_mat - (numpy array) a latent feature by offers matrix
        '''
        self.train_offers = train_df.copy()
        self.val_offers = val_df.copy()

        # create training user item matrix
        train_user_items = train_df[['customer_id', 'offer_id', 'offer_viewed']]
        self.train_user_item_df = train_user_items.groupby(['customer_id', 'offer_id'])['offer_viewed'].max().unstack()
        self.train_user_item_df = self.train_user_item_df.notnull().astype(int)
        self.train_user_item_mat= np.array(self.train_user_item_df)

        # create validation user item matrix
        val_user_items = val_df[['customer_id', 'offer_id', 'offer_viewed']]
        self.val_user_item_df = val_user_items.groupby(['customer_id', 'offer_id'])['offer_viewed'].max().unstack()
        self.val_user_item_df = self.val_user_item_df.notnull().astype(int)
        self.val_user_item_mat= np.array(self.val_user_item_df)        

        # Store more inputs
        self.latent_features = latent_features
        self.learning_rate = learning_rate
        self.iters = iters

        # Set up useful training values to be used through the rest of the function
        self.n_users_train = self.train_user_item_mat.shape[0]  # number of rows in the matrix
        self.n_offers_train = self.train_user_item_mat.shape[1] # number of offers in the matrix
        self.num_ratings_train = np.count_nonzero(~np.isnan(self.train_user_item_mat))  # total number of ratings in the matrix
        self.customer_ids_series_train = np.array(self.train_user_item_df.index)
        self.offer_ids_series_train = np.array(self.train_user_item_df.columns)

        # Set up useful validation values to be used through the rest of the function
        self.n_users_val = self.val_user_item_mat.shape[0]  # number of rows in the matrix
        self.n_offers_val = self.val_user_item_mat.shape[1] # number of offers in the matrix
        self.num_ratings_val = np.count_nonzero(~np.isnan(self.val_user_item_mat))  # total number of ratings in the matrix
        self.customer_ids_series_val = np.array(self.val_user_item_df.index)
        self.offer_ids_series_val = np.array(self.val_user_item_df.columns)        

        # initialize the customer and offer matrices with random values
        # helpful link: https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
        train_user_mat = np.random.rand(self.n_users_train, self.latent_features)   # customer matrix filled with random values of shape customer x latent
        train_offer_mat = np.random.rand(self.latent_features, self.n_offers_train) # offer matrix filled with random values of shape latent x offers

        val_user_mat = np.random.rand(self.n_users_val, self.latent_features)   # customer matrix filled with random values of shape customer x latent
        val_offer_mat = np.random.rand(self.latent_features, self.n_offers_val)

        train_sse_accum = 0 # initialize training sse at 0 for first iteration
        val_sse_accum = 0 # initialize validation sse at 0 for first iteration
        train_ae_accum = 0 # initialize training sse at 0 for first iteration
        val_ae_accum = 0 # initialize validation sse at 0 for first iteration
        train_mse_iter = [] # initialize training MSE iteration list
        val_mse_iter = [] # initialize training MSE iteration list
        train_mae_iter = [] # initialize training MAE iteration list
        val_mae_iter = [] # initialize training MAE iteration list
    
        # keep track of iteration and MSE
        print("Optimization Statistics")
        print(f'{"-" * 25}')

        # for each iteration
        for iteration in range(iters):
        
            # train model
            for i in range(self.n_users_train):
                for j in range(self.n_offers_train):
                
                    # if the rating exists
                    if self.train_user_item_mat[i, j] > 0:
                    
                        # compute the error as the actual minus the dot product of the user and offer latent features
                        diff = self.train_user_item_mat[i, j] - np.dot(train_user_mat[i, :], train_offer_mat[:, j])

                        # Keep track of the sum of squared errors for the matrix
                        train_sse_accum += diff**2
                        train_ae_accum += abs(diff)

                        # update the values in each matrix in the direction of the gradient
                        for k in range(latent_features):
                            train_user_mat[i, k] += learning_rate * (2*diff*train_offer_mat[k, j])
                            train_offer_mat[k, j] += learning_rate * (2*diff*train_user_mat[i, k])

            # validation
            for i in range(self.n_users_val):
                for j in range(self.n_offers_val):
                
                    # if the rating exists
                    if self.val_user_item_mat[i, j] > 0:
                    
                        # compute the error as the actual minus the dot product of the user and offer latent features
                        diff = self.val_user_item_mat[i, j] - np.dot(val_user_mat[i, :], val_offer_mat[:, j])
                        
                        # Keep track of the sum of squared errors for the matrix
                        val_sse_accum += diff**2
                        val_ae_accum += abs(diff)

            # print results every 15 iterations
            if iteration % 15 == 0:
                print(f'Iteration {iteration+1}') 
                print(f'MSE: train = {train_sse_accum / self.num_ratings_train:.4f}, valid. = {val_sse_accum / self.num_ratings_val:.4f}')
                print(f'MAE: train = {train_ae_accum / self.num_ratings_train:.4f}, valid. = {val_ae_accum / self.num_ratings_val:.4f}')
                print(f'{"-" * 35}')
            
            # save mse for plots
            train_mse = train_sse_accum / self.num_ratings_train
            train_mae = train_ae_accum / self.num_ratings_train
            train_mse_iter.append(train_mse)
            train_mae_iter.append(train_mae)
            val_mse = val_sse_accum / self.num_ratings_val
            val_mae = val_ae_accum / self.num_ratings_val
            val_mse_iter.append(val_mse)
            val_mae_iter.append(val_mae)

            # SVD based fit
            # Keep user_mat and offer_mat for safe keeping
            self.train_user_mat = train_user_mat
            self.train_offer_mat = train_offer_mat
            self.val_user_mat = val_user_mat
            self.val_offer_mat = val_offer_mat

            # Knowledge based fit
            self.train_ranked_offers = rf.create_ranked_offers(self.train_offers)
        
        return train_mse_iter, train_mae_iter, val_mse_iter, val_mae_iter
    
    def predict_offer(self, customer_id, offer_id):
        '''
        INPUT:
        customer_id - the customer_id
        offer_id - the offer_id 
    
        OUTPUT:
        pred - the predicted reaction for customer_id / offer_id according to FunkSVD
        '''
        try:
            # customer row and offer column
            customer_row = np.where(self.customer_ids_series_train == customer_id)[0][0]
            offer_col = np.where(self.offer_ids_series_train == offer_id)[0][0]

            # Take dot product of that row and column in U and V to make prediction
            pred = np.dot(self.train_user_mat[customer_row, :], self.train_offer_mat[:, offer_col])

            print('For user {}, we predict a {} rating for offer {}.'.format(customer_id, pred, offer_id))

            return pred

        except:
            print("I'm sorry, but a prediction cannot be made for this user-movie pair.  It looks like one of these items does not exist in our current database.")

            return None
        
    def make_recs(self, _id, _cust_id, rec_num = 5):
        '''
        Input:
            _id - a customer id (int)
            rec_num - number of recommendations to return (int)

        Output:
            recs - (array) a list or numpy array of recommended offers like the 
            given offer, or recs for a customer_id given
        '''

        rec_ids, rec_offers = None, None
        if _id in self.customer_ids_series_train:
            # Get the index of which row the user is in for use in U matrix
            idx = np.where(self.customer_ids_series_train == _id)[0][0]
                
            # take the dot product of that row and the V matrix
            preds = np.dot(self.train_user_mat[idx,:],self.train_offer_mat)
                
            # pull the top movies according to the prediction
            indices = preds.argsort()[-rec_num:][::-1] #indices
            rec_ids = self.offer_ids_series[indices]
            rec_offers = rf.get_offer_ids(rec_ids, self.offers)
            print('Top offers for customer {} according to prediction: {}'.format(_id, rec_offers))
            
        else:
            # if we don't have this user, give just top ratings back
            rec_offers = rf.popular_recommendations(_id, rec_num, self.ranked_offers)
            print("Because this user wasn't in our database, we are giving back the top offer recommendations for all users.")
            print('Top offer recommendations: {}'.format(rec_offers))
        
        return rec_ids, rec_offers
