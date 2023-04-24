# import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

def clean_portfolio():
    '''
    clean_portfolio: 
        - prepares and cleans the Starbucks offer portfolio data for further processing

    In: 
        - None

    Out: portfolio
        - cleaned portfolio pandas dataframe
    '''
    # read in portfolio json data
    portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)

    # update column names
    portfolio.rename(columns={'id' : 'offer_id', 'duration' : 'days_open', 'difficulty' : 'min_spend'},
                      inplace = True)
    
    # initialize MultiLabelBinarizer to one-hot encode channels column
    mlb = MultiLabelBinarizer()
    mlb.fit(portfolio['channels'])
    channels_temp = pd.DataFrame(mlb.transform(portfolio['channels']), columns = mlb.classes_)

    # use pandas get_dummies to one-hot encode offer_type column
    offer_type_temp = pd.get_dummies(portfolio['offer_type'])

    # merge resultant dataframes back into portfolio
    portfolio = pd.concat([portfolio, channels_temp, offer_type_temp], axis = 1)

    # drop channels and offer_type columns
    portfolio.drop(['channels', 'offer_type'], axis = 1, inplace = True)

    # reorder columns
    col_order = ['offer_id', 'reward', 'min_spend', 'days_open', 'email', 
                'mobile', 'social', 'web', 'bogo', 'discount', 'informational']
    
    return portfolio.loc[:, col_order]

def clean_profile():
    '''
    clean_profile: 
        - prepares and cleans the Starbucks customer profile data for further processing

    In:
        - None
    
    Out:
        - cleaned profile pandas dataframe
    '''
    # read in profile json file
    profile = pd.read_json('data/profile.json', orient='records', lines=True)

    # update column names
    profile.rename(columns = {'id' : 'customer_id'}, inplace = True)

    # remove the 2175 rows where "gender" and "income" have missing values and "age" equals 118
    profile = profile[profile.age != 118]

    # create membership_start column from became_member_on columns
    profile['membership_start'] = pd.to_datetime(profile['became_member_on'].astype(str), format='%Y%m%d')

    # create membership_year column from membership_start
    profile['membership_year'] = pd.DatetimeIndex(profile['membership_start']).year

    # one-hot encode 'gender' column
    gender_temp = pd.get_dummies(profile['gender'])
    gender_temp.rename(columns = {'F': 'female', 'M' : 'male', 'O' : 'other'}, inplace = True)

    # merge gender_temp into profile
    profile = pd.concat([profile, gender_temp], axis = 1)

    # drop unneeded columns columns
    profile.drop(['became_member_on', 'gender', 'other'], axis = 1, inplace = True)

    # reorder columns
    col_order = ['customer_id', 'age', 'income', 'female', 'male', 'membership_start', 'membership_year']

    return profile.loc[:,col_order]

def extract_offer_id(value):
    '''
    extract_offer_id:
        - extract offer_id column from transcript dataframe value column
    
    IN:
        - transcript dataframe value colum
    
    OUT:
        - offer_id column extracted from value column where value key is 'offer id' or 'offer_id'
    '''
    if list(value.keys())[0] in ['offer id', 'offer_id']:
        return list(value.values())[0]
    
def extract_amount(value):
    '''
    extract_offer_id:
        - extract amount column from transcript dataframe value column
        
    IN:
        - transcript dataframe value colum
    
    OUT:
        - amount column extracted from value column where value key is 'amount'
    '''
    if list(value.keys())[0] in ['amount']:
        return list(value.values())[0]

def clean_transcript():
    '''
    clean_transcript:
        - prepare and clean transcript data for further processing

    IN:
        - None

    OUT:
        - cleaned transcript pandas dataframe 
    '''
    # read in json datafile
    transcript = pd.read_json('data/transcript.json', orient='records', lines=True)

    # rename 'person' column to 'customer_id'
    transcript.rename(columns = {'person' : 'customer_id'}, inplace = True)

    # one-hot encode 'event' column
    event_temp = pd.get_dummies(transcript['event'])
    event_temp.drop(['transaction'], axis = 1, inplace = True)

    # merge event_temp into transcript
    transcript = pd.concat([transcript, event_temp], axis = 1)
    
    # extract offer_id and amount column from value column
    transcript['offer_id'] = transcript.value.apply(extract_offer_id)
    transcript['amount'] = transcript.value.apply(extract_amount)
    
    # drop event and value
    transcript.drop(['event','value'], axis=1, inplace=True)

    # rename columns
    transcript.rename(columns = {'offer completed' : 'offer_completed', 'offer received' : 'offer_received',
                        'offer viewed' : 'offer_viewed'}, inplace = True)

    # reorder offers columns
    col_order = ['offer_id', 'customer_id', 'offer_completed', 'offer_received', 'offer_viewed', 'amount', 'time']

    return transcript.loc[:, col_order]

def merge_data(portfolio, profile, transcript):
    '''
    merge_data:
        - merge the cleaned datasets
    IN:
        - portfolio_clean - cleaned portfolio data
        - profile_clean - cleaned profile data
        - transcript - cleaned transcript data customer offer and transaction data
    OUT:
        - starbucks_cleaned_data.csv - merged dataset
    '''
    # merge transcript data and portfolio data
    combined_data_df = pd.merge(transcript, portfolio, how = 'left', on = ['offer_id'])

    # merge combined_data_df with profile data
    combined_data_df = pd.merge(combined_data_df, profile, how = 'left', on = ['customer_id'])

    # label encode customer_id and offer_id columns
    label_encoder = LabelEncoder() # initialize LabelEncoder

    combined_data_df['customer_id'] = label_encoder.fit_transform(combined_data_df.customer_id.values)
    combined_data_df['offer_id'] = label_encoder.fit_transform(combined_data_df.offer_id.values)

    # save merged datasets as csv files
    combined_data_df.to_csv('data/starbucks_combined_data.csv', encoding = 'utf-8', index = False)

    return

def age_bins(df):
    '''
    age_bins:
        - create age range bins
    
    IN:
        - dataframe with age column 
    
    OUT:
        - dataframe with age column transformed into age bins
    '''
    age_ranges = pd.cut(df['age'], bins = [18, 29, 44, 59, 74, 89, 101],
                    labels = ['18-29', '30-44', '45-59', '60-74', '75-89', '90-101'])

    # one-hot encode age_ranges
    age_temp = pd.get_dummies(age_ranges)

    # merge age columns into dataframe
    df = pd.concat([df, age_temp], axis = 1)
    
    # drop unneeded age column
    df.drop(['age'], axis = 1, inplace = True)

    return df

def income_bins(df):
    '''
    income_bins:
        - create age range bins
    
    IN:
        - dataframe with income column 
    
    OUT:
        - dataframe with income column transformed into income bins
    '''
    income_ranges = pd.cut(df['income'], bins = [30000, 49000, 69000, 89000, 109000, 120000],
                    labels = ['30k-49k', '50k-69k', '70k-89k', '90k-109k', '110k-120k'])

    # one-hot encode age_ranges
    income_temp = pd.get_dummies(income_ranges)

    # merge age columns into dataframe
    df = pd.concat([df, income_temp], axis = 1)
    
    # drop unneeded age column
    df.drop(['income'], axis = 1, inplace = True)

    return df