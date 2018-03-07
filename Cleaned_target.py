
# coding: utf-8

'''
This Notebook is related to the cleaning and exploration of the target variable
file (POS). The goal of the challenge is to find surroundings to lead to high POS.
This problem can thus be refrased as what are the differences between low and high
POS performances hence a binary classification problem. As such we will explore
the dataset with the goal of finding the timeframe that are as dichotomic as
possible to give the model an easier time at making the distinction between
location with low POS and location with high POS.
'''

__author__ = 'Adriano Vereno'

__status__ = 'Development'
__version__ = '1.0.0'
__date__ = '2018-03-05'


# In[1]:


#We first import the file using the pandas library and make it as a dataframe
import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
target = pd.read_csv('sales_granular.csv')


# In[3]:


print(len(target), len(target.columns))


# There are 906 locations and 11937 features

# In[4]:


target.head()


# Based on the output above, we can see that the features corresponding to hourly
#timestamps, going from August 2015 to June 2017.

# When exploring a dataset, several steps such as exploring the missing values,
#abnormalities, duplicates, values distributions need to be taken.
# These steps don't need to have a specific order so I will first start by
#looking at duplicates.
# If duplicates are present, I will remove them as I want each entry to be unique.

# In[2]:



ids = target.store_code
target[ids.isin(ids[ids.duplicated()])].sort_values("store_code")


# From the above output, we can see that the location with the store code 11028
#has 4 duplicates. I will remove them now.

# In[2]:


target = target.drop_duplicates(keep=False)
len(target)


# The duplicates have been removed. Now let's look at the NAs, from the look of
#the previous outputs, all columns have NAs, however this NAs can be explained
#not by no data available but rather no POS at that time point, meaning 0.
#As such, NAs in this dataset can be filled with 0 which is what will be done now.

# In[3]:


target2 = target.fillna(0, axis=1)
target2.head()


# Now let's look at abnormalities, here data are all numerical and since they
#are Point of Sales, I would expect them to be either 0 or positive. So an
#abnormality would be negative numbers.

# In[14]:


sum(n < 0 for n in target2.values.flatten())


# 170 values are negative, we don't know unfortunately why they were introduced
#and what mistake led to them being introduced.

# In[16]:


pd.set_option('display.max_rows', None)

pd.DataFrame({'min': target2.min(), 'idxmin':target2.idxmin()})


# We don't know how they were introcuded so we will assume that mistakes were
#made and these values should be positive or null so we will change them to 0.

# In[4]:


target2[target2 < 0] = 0


# Now We can have a look at how values are distributed and if outliers are present.
#For that I will split the timeframe by month. The reason comes from the fact
#that yearly might be to restrictive, we may have the bigger picture but might
#missed some underlying changes. On the other hand, a day or week stratification
#might actually be the opposite. Based on that, looking month by month may be a
#balanced view with respect to the other two.

# In[5]:


target_2015 = target2[target2.columns[target2.columns.to_series().str.contains('[0-9]\/[0-9][0-9]\/15')]]
#lets now examine each month. A quick glance of the previous data shows us that
#only the last 5 months have data in 2015.
target_2015_8 = target2[target2.columns[target2.columns.to_series().str.contains('8\/[0-9][0-9]\/15')]]
target_2015_9 = target2[target2.columns[target2.columns.to_series().str.contains('9\/[0-9][0-9]\/15')]]
target_2015_10 = target2[target2.columns[target2.columns.to_series().str.contains('10\/[0-9][0-9]\/15')]]
target_2015_11 = target2[target2.columns[target2.columns.to_series().str.contains('11\/[0-9][0-9]\/15')]]
target_2015_12 = target2[target2.columns[target2.columns.to_series().str.contains('12\/[0-9][0-9]\/15')]]

target_response_2015_8 = target_2015_8.sum(1)
target_response_2015_9 = target_2015_9.sum(1)
target_response_2015_10 = target_2015_10.sum(1)
target_response_2015_11 = target_2015_11.sum(1)
target_response_2015_12 = target_2015_12.sum(1)

#In this cell, we regex all month from 2015 and add each column to end up with
#the total POS per month for each location.
#In the next cell, we will have a look at the values distribution by using boxplots.


# In[20]:


plt.boxplot([target_response_2015_8, target_response_2015_9, target_response_2015_10, target_response_2015_11, target_response_2015_12])


# As we can see, we cannot really do much because most of the values are actually
#outliers. So this visualisation is pretty much useless. We will try to have a
#look then using bar charts to have a better idea of the values distribution.

# In[23]:

fig = plt.figure(figsize=(10, 10))
sub1 = fig.add_subplot(321)
sub1.set_title('Distribution of POS 8/2015')
sub1.hist(target_response_2015_8, range=(0, 165750))
sub2 = fig.add_subplot(322, axisbg="lightgrey")
sub2.set_title('Distribution of POS 9/2015')
sub2.hist(target_response_2015_9, range=(0, 165750))
sub3 = fig.add_subplot(323)
sub3.set_title('Distribution of POS 10/2015')
sub3.hist(target_response_2015_10, range=(0, 165750))
sub4 = fig.add_subplot(324, axisbg="lightgrey")
sub4.set_title('Distribution of POS 11/2015')
sub4.hist(target_response_2015_11, range=(0, 165750))
sub5 = fig.add_subplot(325, axisbg="lightgrey")
sub5.set_title('Distribution of POS 12/2015')
sub5.hist(target_response_2015_12, range=(0, 165750))
plt.show()


# This is not better, as most of the bins are between 0 and 20000 thousand POS,
#but we can see that we have outliers.
# Let's look now at the numerical description to have a better idea of the
#maximum and minimum.

# In[43]:


pd.concat([target_response_2015_8.describe(), target_response_2015_9.describe(), target_response_2015_10.describe(), target_response_2015_11.describe(), target_response_2015_12.describe()], axis=1)


#In the output above, we can that there are huge differences between values due
#to the high amount of outliers. A solution could then be to take the month with 
#the lowest standard deviation, as the effect of outliers would be lessen. For 
#that the first month of 2015 could be really usefull, as it has smallest 
#standard deviation.
#Now Let's have a look at 2016 and 2017, since the goal would be to eventually 
#find the months maximizing the differences.

# In[6]:


target_2016 = target2[target2.columns[target2.columns.to_series().str.contains('[0-9]\/[0-9][0-9]\/16')]]

target_2016_1 = target2[target2.columns[target2.columns.to_series().str.contains('^[1]{1}\/[0-9][0-9]\/16')]]
target_2016_2 = target2[target2.columns[target2.columns.to_series().str.contains('^2\/[0-9][0-9]\/16')]]
target_2016_3 = target2[target2.columns[target2.columns.to_series().str.contains('3\/[0-9][0-9]\/16')]]
target_2016_4 = target2[target2.columns[target2.columns.to_series().str.contains('4\/[0-9][0-9]\/16')]]
target_2016_5 = target2[target2.columns[target2.columns.to_series().str.contains('5\/[0-9][0-9]\/16')]]
target_2016_6 = target2[target2.columns[target2.columns.to_series().str.contains('6\/[0-9][0-9]\/16')]]
target_2016_7 = target2[target2.columns[target2.columns.to_series().str.contains('7\/[0-9][0-9]\/16')]]
target_2016_8 = target2[target2.columns[target2.columns.to_series().str.contains('8\/[0-9][0-9]\/16')]]
target_2016_9 = target2[target2.columns[target2.columns.to_series().str.contains('9\/[0-9][0-9]\/16')]]
target_2016_10 = target2[target2.columns[target2.columns.to_series().str.contains('10\/[0-9][0-9]\/16')]]
target_2016_11 = target2[target2.columns[target2.columns.to_series().str.contains('11\/[0-9][0-9]\/16')]]
target_2016_12 = target2[target2.columns[target2.columns.to_series().str.contains('12\/[0-9][0-9]\/16')]]

target_response_2016_1 = target_2016_1.sum(1)
target_response_2016_2 = target_2016_2.sum(1)
target_response_2016_3 = target_2016_3.sum(1)
target_response_2016_4 = target_2016_4.sum(1)
target_response_2016_5 = target_2016_5.sum(1)
target_response_2016_6 = target_2016_6.sum(1)
target_response_2016_7 = target_2016_7.sum(1)
target_response_2016_8 = target_2016_8.sum(1)
target_response_2016_9 = target_2016_9.sum(1)
target_response_2016_10 = target_2016_10.sum(1)
target_response_2016_11 = target_2016_11.sum(1)
target_response_2016_12 = target_2016_12.sum(1)

fig = plt.figure(figsize=(15, 15))
sub1 = fig.add_subplot(621)
sub1.set_title('Distribution of POS 1/2016')
sub1.hist(target_response_2016_1, range=(0, 150000))
sub2 = fig.add_subplot(622, axisbg="lightgrey")
sub2.set_title('Distribution of POS 2/2016')
sub2.hist(target_response_2016_2, range=(0, 150000))
sub3 = fig.add_subplot(623)
sub3.set_title('Distribution of POS 3/2016')
sub3.hist(target_response_2016_3, range=(0, 150000))
sub4 = fig.add_subplot(624, axisbg="lightgrey")
sub4.set_title('Distribution of POS 4/2016')
sub4.hist(target_response_2016_4, range=(0, 150000))
sub5 = fig.add_subplot(625, axisbg="lightgrey")
sub5.set_title('Distribution of POS 5/2016')
sub5.hist(target_response_2016_5, range=(0, 150000))
sub6 = fig.add_subplot(626)
sub6.set_title('Distribution of POS 6/2016')
sub6.hist(target_response_2016_6, range=(0, 150000))
sub7 = fig.add_subplot(627, axisbg="lightgrey")
sub7.set_title('Distribution of POS 7/2016')
sub7.hist(target_response_2016_7, range=(0, 150000))
sub8 = fig.add_subplot(628)
sub8.set_title('Distribution of POS 8/2016')
sub8.hist(target_response_2016_8, range=(0, 150000))
sub9 = fig.add_subplot(629, axisbg="lightgrey")
sub9.set_title('Distribution of POS 9/2016')
sub9.hist(target_response_2016_9, range=(0, 150000))
sub10 = fig.add_subplot(631, axisbg="lightgrey")
sub10.set_title('Distribution of POS 10/2016')
sub10.hist(target_response_2016_10, range=(0, 150000))
sub11 = fig.add_subplot(632)
sub11.set_title('Distribution of POS 11/2016')
sub11.hist(target_response_2016_11, range=(0, 150000))
sub12 = fig.add_subplot(633, axisbg="lightgrey")
sub12.set_title('Distribution of POS 12/2016')
sub12.hist(target_response_2016_12, range=(0, 150000))

plt.show()


# In[7]:


target_2017 = target2[target2.columns[target2.columns.to_series().str.contains('[0-9]\/[0-9][0-9]\/17')]]


target_2017_1 = target2[target2.columns[target2.columns.to_series().str.contains('1\/[0-9][0-9]\/17')]]
target_2017_2 = target2[target2.columns[target2.columns.to_series().str.contains('2\/[0-9][0-9]\/17')]]
target_2017_3 = target2[target2.columns[target2.columns.to_series().str.contains('3\/[0-9][0-9]\/17')]]
target_2017_4 = target2[target2.columns[target2.columns.to_series().str.contains('4\/[0-9][0-9]\/17')]]
target_2017_5 = target2[target2.columns[target2.columns.to_series().str.contains('5\/[0-9][0-9]\/17')]]
target_2017_6 = target2[target2.columns[target2.columns.to_series().str.contains('6\/[0-9][0-9]\/17')]]


target_response_2017_1 = target_2017_1.sum(1)
target_response_2017_2 = target_2017_2.sum(1)
target_response_2017_3 = target_2017_3.sum(1)
target_response_2017_4 = target_2017_4.sum(1)
target_response_2017_5 = target_2017_5.sum(1)
target_response_2017_6 = target_2017_6.sum(1)


fig = plt.figure(figsize=(10, 10))
sub1 = fig.add_subplot(421)
sub1.set_title('Distribution of POS 1/2017')
sub1.hist(target_response_2017_1, range=(0, 150000))
sub2 = fig.add_subplot(422, axisbg="lightgrey")
sub2.set_title('Distribution of POS 2/2017')
sub2.hist(target_response_2017_2, range=(0, 150000))
sub3 = fig.add_subplot(423)
sub3.set_title('Distribution of POS 3/2017')
sub3.hist(target_response_2017_3, range=(0, 150000))
sub4 = fig.add_subplot(424, axisbg="lightgrey")
sub4.set_title('Distribution of POS 4/2017')
sub4.hist(target_response_2017_4, range=(0, 150000))
sub5 = fig.add_subplot(425, axisbg="lightgrey")
sub5.set_title('Distribution of POS 5/2017')
sub5.hist(target_response_2017_5, range=(0, 150000))
sub6 = fig.add_subplot(426, axisbg="lightgrey")
sub6.set_title('Distribution of POS 6/2017')
sub6.hist(target_response_2017_6, range=(0, 150000))

plt.show()


#This is a bit quick and dirty but we can see that 2016 and 2017 have similar
#distributions as 2015, with lots of outliers, so I expect the boxplot to be
#also similar, hence I will go directy to the numerical description.

# In[28]:


pd.concat([target_response_2016_1.describe(), target_response_2016_2.describe(), target_response_2016_3.describe(), target_response_2016_4.describe(), target_response_2016_5.describe(), target_response_2016_6.describe(), target_response_2016_7.describe(), target_response_2016_8.describe(), target_response_2016_9.describe(), target_response_2016_10.describe(), target_response_2016_11.describe(), target_response_2016_12.describe()], axis=1)


# In[48]:


print([target_response_2016_1.median(), target_response_2016_2.median(), target_response_2016_3.median(), target_response_2016_4.median(), target_response_2016_5.median(), target_response_2016_6.median(), target_response_2016_7.median(), target_response_2016_8.median(), target_response_2016_9.median(), target_response_2016_10.median(), target_response_2016_11.median(), target_response_2016_12.median()])


#Based on the standard deviasions, the first month of 2016 seems really good.
# In[29]:


pd.concat([target_response_2017_1.describe(), target_response_2017_2.describe(), target_response_2017_3.describe(), target_response_2017_4.describe(), target_response_2017_5.describe(), target_response_2017_6.describe()], axis=1)


# In[49]:


print([target_response_2017_1.median(), target_response_2017_2.median(), target_response_2017_3.median(), target_response_2017_4.median(), target_response_2017_5.median(), target_response_2017_6.median()])


#Here, the last month of 2017 seems to be the best one in term of standard deviation.
#Now what is left to do is concatenate and sum up these three month into one
#single dataframe.


# In[8]:

target_response_2015_8_2016_1_2017_6 = pd.concat([target_response_2015_8, target_response_2016_1, target_response_2017_6], axis=1)
target_response_2015_8_2016_1_2017_6 = target_response_2015_8_2016_1_2017_6.sum(1)
target_response_2015_8_2016_1_2017_6 = pd.concat([target2.store_code, target_response_2015_8_2016_1_2017_6], axis=1)
target_response_2015_8_2016_1_2017_6.columns = ['store_code', 'POS_binary']
target_response_2015_8_2016_1_2017_6.to_csv('target_response_2015_8_2016_1_2017_6.csv')

#The exploration of the target is done and has been saved as csv. I will not
#make the binary conversion yet because I want to wait after merging it with
#the exploratory variables. Once all the cleaning steps have been done on the
#merged dataset, I will proceed to binarize the response variable.
