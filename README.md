# Udacity Capstone Project
## Starbucks Promotional Offers
#### Marc Mascaro
#### May 2023

# Contents
* [Introduction](#introduction)
* [Python Version and Libraries Used](#python-version-and-libraries-used)
* [Files in Repository](#files-in-repository)
* [References](#references)
* [Acknowledgements](#acknowledgements)

## Introduction
This project is being undertaken as the final project in the [Udacity Data Scientist Nanodegree Program](https://www.udacity.com/course/data-scientist-nanodegree--nd025).

The Starbucks data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks.

Not all users receive the same offer, and that is the challenge to solve with this data set.

The task is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.

Every offer has a validity period before the offer expires. As an example, a BOGO offer might be valid for only 5 days. You'll see in the data set that informational offers have a validity period even though these ads are merely providing information about a product; for example, if an informational offer has 7 days of validity, you can assume the customer is feeling the influence of the offer for 7 days after receiving the advertisement.

## Python Version and Libraries Used
**Python Version:** 3.9.16 

The following standard libraries were used:
- pandas
- numpy
- matplotlib
- seaborn
- math
- json
- sklearn
  - MultiLabelBinarizer
  - LabelEncoder
  - train_test_split

The following self-written libraries were used:
- wrangle_data.py - provides data cleaning functions
- recommender.py - Recommender class and associated functions
- recommender_functions.py - additional functions associated with recommender

## Files in Repository
The following files are in the repository:

| File | Description |
|------|-------------|
| coffee_cups.jpg | Picture of two Starbucks coffee mugs taken in my kitchen |
| recommender.py | Sets up Recommender class that fits FunkSVD model, predicts offers, and makes recommendations |
| recommender_functions.py | Companion functions to Recommender |
| starbucks_capstone.ipynb | Jupyter notebook that drives the analysis of the Starbucks data |
| wrangle_data.py | Functions to clean and combine source data |
| data/portfolio.json | Starbucks offer portfolio event data
| data/profile.json | Starbucks rewards customer data
| data/starbucks_combined_data.csv | Cleaned, combined Starbucks dataset
| data/transcript.json | Starbucks event log |

## References

| Reference Item      | URL                         |
|---------------------|-----------------------------|
| MultiLabelBinarizer | https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html
| LabelEncoder        | http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html, https://stackoverflow.com/questions/50258960/how-to-apply-labelencoder-for-a-specific-column-in-pandas-dataframe, https://www.geeksforgeeks.org/ml-label-encoding-of-datasets-in-python/
| Change column order in a dataframe | https://www.geeksforgeeks.org/change-the-order-of-a-pandas-dataframe-columns-in-python/
| Delete rows of dataframe based on column value | https://stackoverflow.com/questions/18172851/deleting-dataframe-row-in-pandas-based-on-column-value
| Extract year from datetime column | https://stackoverflow.com/questions/25146121/extracting-just-month-and-year-separately-from-pandas-datetime-column
| Pandas cut() method | https://www.geeksforgeeks.org/pandas-cut-method-in-python/
| Pandas groupby with as_index | https://stackoverflow.com/questions/41236370/what-is-as-index-in-groupby-in-pandas
| Address chained index errors | https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
| Manhattan distance | https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html, https://stackoverflow.com/questions/47736531/vectorized-matrix-manhattan-distance-in-numpy
| Pandas .at() | https://www.geeksforgeeks.org/python-pandas-dataframe-at/
| Create dataframe from multiple numpy arrays | https://stackoverflow.com/questions/49263247/how-can-i-make-a-pandas-dataframe-out-of-multiple-numpy-arrays
| Train/Test/Split usage | https://www.geeksforgeeks.org/how-to-split-the-dataset-with-scikit-learns-train_test_split-function/
| Create dataframe from dataframe .describe() | https://stackoverflow.com/questions/34026089/create-dataframe-from-another-dataframe-describe-pandas
              

## Acknowledgements
[Udacity](https://www.udacity.com/) and [Starbucks](https://www.starbucks.com/) - for providing the datasets for this project.
