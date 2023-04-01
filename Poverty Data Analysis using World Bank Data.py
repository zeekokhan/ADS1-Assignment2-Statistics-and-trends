#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

def read_datafiles(path):
    """
    The read_datafiles function reads in Worldbank data from a given path and returns two pandas dataframes, one with years as columns and another with countries as columns.

    Parameters:
        path (str): The path to the directory containing the Worldbank data files.
    
    Returns:
        countries_df (pandas dataframe): A dataframe with countries as columns and indicators as rows.
        years_df (pandas dataframe): A dataframe with years as columns and indicators as rows.
    """
    
    # Read in the Worldbank data
    df = None
    for i in os.listdir(path):
        data_path = os.path.join(path, i)
        temp = pd.read_csv(data_path, skiprows=4, index_col=0)
        if df is not None:
            df = pd.concat([df, temp], axis=0)
        else:
            df = temp
    
    df.drop("Unnamed: 66", axis=1, inplace=True)
    
    
    # Transpose the dataframe to create one with years as columns
    years_df = df.T
    
    # Transpose the dataframe again to create one with countries as columns
    countries_df = years_df.T
    
    # Clean the data
    countries_df.columns.name = ''
    
    return countries_df, years_df


# In[2]:


# Read in the data
years_df, countries_df = read_datafiles('poverty_dataset')

# Print the first few rows of the years dataframe
years_df.head()


# In[3]:


# Print the first few rows of the countries dataframe
countries_df.head()


# In[4]:


# Top 2 Poorest Countries
countries = ['Burundi',
            'Central African Republic']

# Get summary statistics for the population indicator for the top two poorest countries
temp = years_df.reset_index()
temp = temp[temp["Country Name"].isin(countries)].set_index(["Country Name", "Indicator Name"]).drop(["Country Code", "Indicator Code"], axis=1)
temp = temp.T.astype(np.number).describe().sort_index(axis=1)
display(temp)


# In[5]:


indicator = "Population, total"

temp = years_df.reset_index()
temp = temp[temp["Country Name"].isin(countries)]
temp = temp[temp["Indicator Name"] == indicator].drop(["Country Code", "Indicator Code", "Indicator Name"], axis=1).set_index("Country Name")
temp.T.plot(kind="line", figsize=(10, 4))
plt.title("Population of Countries by Year")
plt.ylabel("Population Total")
plt.xlabel("Year")
plt.legend()
plt.show()

