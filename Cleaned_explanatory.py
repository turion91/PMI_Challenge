
# coding: utf-8


'''
This Notebook is related to the cleaning of the feature dataset. The feature
file was originally as a json file, but for simplicity of exploration and analysis,
I converted it into a csv using JSON2CSV library. Credit for this library goes to
https://github.com/vladikk/json2csv
'''

__author__ = 'Adriano Vereno'

__status__ = 'Development'
__version__ = '1.0.0'
__date__ = '2018-03-05'

# In[ ]:


#import csv
import pandas as pd
data = pd.read_csv('output.csv')
#We have a lot of NA, it is impossible to explore all the 160000+ columns so
#I will trim it, remove all the columns with more than 90% of missing values
#and save it as a new csv file. This part was done on the shell terminal to
#reduce memory consumption.
data2 = data.dropna(thresh=len(data) - 50, axis=1)
data2.to_csv('output_clean_NA_2.csv')


#Now that things are a bit clearer, let's open the new csv and start working.
#Again like the target, we will first look at the duplicates, the remaning NAs,
#then we will proceed by looking at what variables to drop based on logic, as
#well as correlations.

# In[9]:

data = pd.read_csv('output_clean_NA_2.csv')


# In[11]:


print(len(data), len(data.columns))


#This is interesting, we only have 546 store codes, while the target file has
#around 900. This means that almost half of the target locations will not have
#features and thus will be useless. So when merging the two dataset, only 546
#entries will be able to be used. As for the columns, the previous NA removal
#was quite drastic as only 169 columns remane, which still way too much and
#will have to be trimmed down.

# In[12]:


ids = data.store_code
data[ids.isin(ids[ids.duplicated()])].sort_values("store_code")


#Like in the target file, the store_code 11028 has duplicates which will be removed.

# In[13]:


data = data.drop_duplicates(keep=False)


# In[14]:


data.isnull().values.sum()


#We still have a lot of entries with missing values, here in this case, the
#reason for these missing values is really because there are simply no information
#about it. Since each location/surrounding area is unique and are location based,
#I would prefer to remove the columns with again too many NAs rather than
#imputing these values based on the column mode or median, as it may generate
#wrong fakes data that may cause problems to our model.

# In[15]:


def missing_values_table(df):
    '''
    This function is used to count and return the percentage of NAs for each columns.
    '''
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0:'Missing Values', 1:'% of Total Values'})
    return mis_val_table_ren_columns
missing_values_table(data)


# In[17]:


data2 = data.drop_duplicates(keep=False)


# In[18]:


len(data2)


# In[19]:


data2 = data2.drop('Unnamed: 0', axis=1)
#This column means nothing


# In[ ]:


data2.to_csv('output_extra_clean_10_percent.csv')


#The export as csv concludes this notebook, the next steps will happen in the
#Merged_dataframe.py.
