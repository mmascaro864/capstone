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
        
    def fit(self, df, latent_features=4, learning_rate=0.0001, iters=100):
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
            movie_mat - (numpy array) a latent feature by movie matrix
        '''
        self.offers = df.copy()

        # create user item matrix
        user_items = df[['customer_id', 'offer_id', 'offer_viewed']]
        self.user_item_df = user_items.groupby(['customer_id', 'offer_id'])['offer_viewed'].max().unstack()
        #self.user_item_df = user_item_matrix.notnull().astype(int)
        self.user_item_mat= np.array(self.user_item_df)

        # Store more inputs
        self.latent_features = latent_features
        self.learning_rate = learning_rate
        self.iters = iters

        # Set up useful values to be used through the rest of the function
        self.n_users = self.user_item_mat.shape[0]  # number of rows in the matrix
        self.n_offers = self.user_item_mat.shape[1] # number of movies in the matrix
        self.num_ratings = np.count_nonzero(~np.isnan(self.user_item_mat))  # total number of ratings in the matrix
        self.user_ids_series = np.array(self.user_item_df.index)
        self.movie_ids_series = np.array(self.user_item_df.columns)

        # initialize the user and movie matrices with random values
        # helpful link: https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
        user_mat = np.random.rand(self.n_users, self.latent_features)   # customer matrix filled with random values of shape customer x latent
        offer_mat = np.random.rand(self.latent_features, self.n_offers) # offer matrix filled with random values of shape latent x offers

        sse_accum = 0 # initialize sse at 0 for first iteration
        mse_iter = [] # initialize MSE iteration list
    
        # keep track of iteration and MSE
        print("Optimization Statistics")
        print("Iterations | Mean Squared Error ")

        # for each iteration
        for iteration in range(iters):

            # update our sse
            old_sse = sse_accum
            sse_accum = 0
        
            # For each user-offer pair
            for i in range(self.n_users):
                for j in range(self.n_offers):
                
                    # if the rating exists
                    if self.user_item_mat[i, j] > 0:
                    
                        # compute the error as the actual minus the dot product of the user and offer latent features
                        diff = self.user_item_mat[i, j] - np.dot(user_mat[i, :], offer_mat[:, j])

                        # Keep track of the sum of squared errors for the matrix
                        sse_accum += diff**2

                        # update the values in each matrix in the direction of the gradient
                        for k in range(latent_features):
                            user_mat[i, k] += learning_rate * (2*diff*offer_mat[k, j])
                            offer_mat[k, j] += learning_rate * (2*diff*user_mat[i, k])

            # print results every 15 iterations
            if iteration % 15 == 0:
                print("%d \t\t %f" % (iteration+1, sse_accum / self.num_ratings))
            
            # save mse for plots
            mse = sse_accum / self.num_ratings
            mse_iter.append(mse)

            # SVD based fit
            # Keep user_mat and movie_mat for safe keeping
            self.user_mat = user_mat
            self.movie_mat = offer_mat

            # Knowledge based fit
            self.ranked_offers = rf.create_ranked_offers(self.offers)
        
        return user_mat, offer_mat, mse_iter