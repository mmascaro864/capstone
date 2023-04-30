# import necessary libraries
import pandas as pd
import numpy as np

def create_ranked_offers(df):
    '''
    create_ranked_offers:
        - determine top ranked offers based on offers_viewed vs. offers_received
    
    INPUT:
        - df: dataframe containing offers data
       
    OUTPUT:
        - ranked_completed: the ranked viewed offers
    '''
    
    offers = df.groupby('offer_id').sum().reset_index()
    
    ranked_viewed = offers.loc[:,['offer_id', 'offer_received', 'offer_viewed']]
    ranked_viewed['viewed_ratio'] = ranked_viewed['offer_viewed'] / ranked_viewed['offer_received']
    
    ranked_viewed = ranked_viewed.sort_values(['viewed_ratio'], ascending = False)
    
    return ranked_viewed

def popular_recommendations(ranked_viewed, n_top):
    '''
    popular_recommendations:
        - return a ranked list
    
    INPUT:
        - ranked_completed - a dataframe of the ranked offer ids representing offers viewed and received
        - n_top - an integer of the number recommendations you want back 

    OUTPUT:
        - top_offers - a list of the n_top recommended offers
    '''
    # set max_offers equal to number of rows
    max_offers = ranked_viewed.shape[0]
  
    if n_top <= max_offers:
        top_offers = list(ranked_viewed['offer_id'][:n_top])
        print('The top ', n_top, ' offer recommendations: ')
    else:
        return print('Please enter a value less than or equal to {}'.format(max_offers))

    return top_offers

def create_user_item_matrix(df):
    '''
    INPUT:
        df - pandas dataframe with customer_id, offer_id, and offer_viewed columns
    
    OUTPUT:
        user_item - user item matrix 
    
    Description:
        Return a matrix with customer ids as rows and offer ids on the columns with 1 values where a user viewed 
        an offer and a 0 otherwise
    '''
    user_items = df[['customer_id', 'offer_id', 'offer_viewed']]
    user_item_matrix = user_items.groupby(['customer_id', 'offer_id'])['offer_viewed'].max().unstack()
    user_item_matrix = user_item_matrix.notnull().astype(int)
    
    return user_item_matrix # return the user_item matrix 

def find_similar_users(customer_id, user_item):
    '''
    INPUT:
        user_id - (int) a user_id
        user_item - (pandas dataframe) matrix of customer_id by offer_id: 
                1's when a user has interacted with an offer, 0 otherwise
    
    OUTPUT:
        similar_users - (list) an ordered list where the closest users (largest dot product users)
                        are listed first
    
    Description:
        - Computes the similarity of every pair of users based on the dot product
        Returns an ordered
    
    '''
    # compute similarity of each user to the provided user
    user_sim = user_item.dot(user_item.loc[customer_id])

    # sort by similarity
    #user_sim = user_sim[user_id].sort_values(ascending = False)
    user_sim = pd.Series(data = user_sim, index = user_item.index).sort_values(ascending = False)

    # create list of just the ids
    most_similar_users = list(user_sim.index)
   
    # remove the own user's id
    most_similar_users.remove(customer_id)
       
    return most_similar_users # return a list of the users in order from most to least similar