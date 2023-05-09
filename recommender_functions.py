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

def popular_recommendations_old(ranked_viewed, n_top):
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

def popular_recommendations(customer_id, n_top, ranked_viewed):
    '''
    INPUT:
        customer_id - the customer_id (str) of the individual you are making recommendations for
        n_top - an integer of the number recommendations you want back
        ranked_viewed - a dataframe of the ranked offer ids representing offers viewed and received

    OUTPUT:
        top_offers - a list of the n_top recommended offers
    '''
    # set max_offers equal to number of rows
    max_offers = ranked_viewed.shape[0]
    
    if n_top <= max_offers:
        top_offers = list(ranked_viewed['offer_id'][:n_top])
        print('The top {} offer recommendations for customer {}:'.format(n_top, customer_id))
    else:
        return print('Please enter an offer_id value less than or equal to {}.'.format(max_offers))

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

def get_offer_ids(offer_ids, df):
    '''
    INPUT:
    offer_id - (list) a list of offer ids
    df - (pandas dataframe) merged combined_df after cleaning original dataframes
    
    OUTPUT:
    offer_ids - a list of Starbucks offer ids 
    '''
    # create empty list
    offer_ids_list = []
        
    # loop to get offer_ids
    for id in offer_ids_list:
        offer_ids.append(df[df['offer_id'] == float(id)].iloc[0])
    
    return offer_ids # Return the offer_ids

def get_user_offers(customer_id, user_item):
    '''
    INPUT:
    user_id - (int) a user id
    user_item - (pandas dataframe) matrix of customer_ids by offer_ids: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    offer_ids - (list) a list of the offer ids seen by the customer
    
    Description:
    Provides a list of the offer_ids that have been seen by a customer
    '''
    offer_ids = list(user_item.loc[customer_id][user_item.loc[customer_id] == 1].index.astype(str))
    
    return offer_ids # return the ids and names

def user_user_recs(customer_id, user_item_matrix, m = 10):
    '''
    INPUT:
    customer_id - (int) a customer id
    m - (int) the number of recommendations you want for the customer
    
    OUTPUT:
    recs - (list) a list of recommendations for the customer
    
    Description:
    Loops through the users based on closeness to the input customer_id
    For each customer - finds offer_ids the customer hasn't seen before and provides them as recs
    Does this until m recommendations are found
    
    Notes:
    Customers who are the same closeness are chosen arbitrarily as the 'next' user
    
    For the customer where the number of recommended offers starts below m 
    and ends exceeding m, the last items are chosen arbitrarily
    
    '''
    # offers seen by customer
    offers_seen = get_user_offers(customer_id, user_item_matrix)[0]
    similar_users = find_similar_users(customer_id, user_item_matrix)
    
    # keep recommended offers here
    recs = np.array([])
    
    # Go through the neighbors and identify offer_ids they like the customer hasn't seen
    for user in similar_users:
        user_likes = get_user_offers(user, user_item_matrix)[0]
        
        # Obtain recommendations for each neighbor
        new_recs = np.setdiff1d(user_likes, offers_seen, assume_unique=True)
        
        # Update recs with new recs
        recs = np.unique(np.concatenate([new_recs, recs], axis=0))
        
        # If we have enough recommendations exit the loop
        if len(recs) > m-1:
            break
    
    recs = recs[:m] 
    
    return recs # return your recommendations for this customer_id   

def compute_correlation(user_item_matrix, user1, user2):
    '''
    INPUT
        user1 - int customer_id
        user2 - int customer_id
    OUTPUT
        corr - the correlation between the matching offers between the two users
    '''
    # create dataframe from users
    corr_df = user_item_matrix.loc[[user1, user2]]
    
    # calculate correlation
    corr = corr_df.transpose().corr().iloc[0,1]
    
    return corr # return correlation 

def compute_euclidean_dist(user_item_matrix, user1, user2):
    '''
    INPUT
        user1 - int customer_id
        user2 - int customer_id
    OUTPUT
        dist - the euclidean distance between user1 and user2
    '''
    # get customer data
    cust1 = user_item_matrix.loc[user1]
    cust2 = user_item_matrix.loc[user2]
    
    # compute the difference
    diff = cust1 - cust2
    
    # drop missing values
    diff = diff.dropna()
    
    #calculate the norm
    dist = np.linalg.norm(diff)
    
    return dist #return the correlation

def compute_manhattan_dist(user_item_matrix, user1, user2):
    '''
    INPUT
        user1 - int customer_id
        user2 - int customer_id
    OUTPUT
        dist - the manhattan distance between user1 and user2
    '''
    from scipy.spatial import distance
    
    # create dataframe from users
    df = user_item_matrix.loc[[user1, user2]]
    
    # Calculate Manhattan distance between the users
    dist = distance.cityblock(df.loc[user1], df.loc[user2])
    
    return dist #return the manhattan distance