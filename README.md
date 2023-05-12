# Udacity Capstone Project
## Starbucks Promotional Offers
#### Marc Mascaro
#### May 2023

# Contents
* [Project Motivation](#project-motivation)
* [Python Version and Libraries Used](#python-version-and-libraries-used)
* [Files in Repository](#files-in-repository)
* [References](#references)
* [Acknowledgements](#acknowledgements)

## Project Motivation

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
- wrangle_data - provides data cleaning functions
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

              

## Acknowledgements
