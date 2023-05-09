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
        self.customer_ids_series = np.array(self.user_item_df.index)
        self.offer_ids_series = np.array(self.user_item_df.columns)

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
            self.offer_mat = offer_mat

            # Knowledge based fit
            self.ranked_offers = rf.create_ranked_offers(self.offers)
        
        return user_mat, offer_mat, mse_iter
    
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
            customer_row = np.where(self.customer_ids_series == customer_id)[0][0]
            offer_col = np.where(self.offer_ids_series == offer_id)[0][0]

            # Take dot product of that row and column in U and V to make prediction
            pred = np.dot(self.user_mat[customer_row, :], self.offer_mat[:, offer_col])

            #offer_name = str(self.offers[self.offers['offer_id'] == offer_id]['offer_type']) [5:]
            #offer_name = offer_name.replace('\nName: movie, dtype: object', '')
            #print("For user {} we predict a {} rating for the movie {}.".format(customer_id, round(pred, 2), str(offer_name)))

            return pred

        except:
            print("I'm sorry, but a prediction cannot be made for this user-movie pair.  It looks like one of these items does not exist in our current database.")

            return None
        
    def make_recs(self, _id, _id_type='offer', rec_num = 5):
        '''
        Input:
            _id - either a user or movie id (int)
            _id_type - "movie" or "user" (str)
            rec_num - number of recommendations to return (int)

        Output:
            recs - (array) a list or numpy array of recommended movies like the 
            given movie, or recs for a user_id given
        '''

        rec_ids, rec_offers = None, None
        if _id_type == 'customer':
            if _id in self.customer_ids_series:
                # Get the index of which row the user is in for use in U matrix
                idx = np.where(self.customer_ids_series == _id)[0][0]
                
                # take the dot product of that row and the V matrix
                preds = np.dot(self.user_mat[idx,:],self.offer_mat)
                
                # pull the top movies according to the prediction
                indices = preds.argsort()[-rec_num:][::-1] #indices
                rec_ids = self.offer_ids_series[indices]
                rec_offers = rf.get_offer_ids(rec_ids, self.offers)
                print('Top offers according to prediction: {}'.format(rec_offers))
            
            else:
                # if we don't have this user, give just top ratings back
                rec_offers = rf.popular_recommendations(_id, rec_num, self.ranked_offers)
                print("Because this user wasn't in our database, we are giving back the top offer recommendations for all users.")
                print('Top offer recommendations: {}'.format(rec_offers))
        
            
        # Find similar movies if it is a movie that is passed
        else:
            if _id in self.offer_ids_series:
                rec_offers = list(rf.find_similar_users(_id, self.offer_mat))[:rec_num]
            else:
                print("That offer doesn't exist in our database.\nSorry, we don't have any recommendations for you.")
    
        return rec_ids, rec_offers
