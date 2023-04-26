# import necessary libraries
import pandas as pd
import numpy as np

def create_ranked_offers(df):
    '''
    create_ranked_offers:
        - determine top ranked offers based on offers_completed vs. offers_viewed
    
    INPUT:
        - df: dataframe containing offers data
       
    OUTPUT:
        - ranked_completed: the ranked completed offers
    '''
    
    offers = df.groupby('offer_id').sum().reset_index()
    
    ranked_completed = offers.loc[:,['offer_id','offer_viewed', 'offer_completed']]
    ranked_completed['completion_ratio'] = ranked_completed['offer_completed'] / ranked_completed['offer_viewed']
    
    ranked_completed = ranked_completed.sort_values(['completion_ratio'], ascending = False)
    
    return ranked_completed

def popular_recommendations(ranked_completed, n_top):
    '''
    popular_recommendations:
        - return a ranked list
    INPUT:
    ranked_completed - a dataframe of the ranked offer ids representing offers viewed and completed
    n_top - an integer of the number recommendations you want back 

    OUTPUT:
    top_offers - a list of the n_top recommended offers
    '''
    # set max_offers equal to number of rows
    max_offers = ranked_completed.shape[0]
  
    if n_top <= max_offers:
        top_offers = list(ranked_completed['offer_id'][:n_top])
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
    Return a matrix with customer ids as rows and offer ids on the columns with 1 values where a user interacted with 
    an offer and a 0 otherwise
    '''
    user_items = df[['customer_id', 'offer_id', 'offer_viewed']]
    user_item_matrix = user_items.groupby(['customer_id', 'offer_id'])['offer_viewed'].max().unstack()
    user_item_matrix = user_item_matrix.notnull().astype(int)
    
    return user_item_matrix # return the user_item matrix 