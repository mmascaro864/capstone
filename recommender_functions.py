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

