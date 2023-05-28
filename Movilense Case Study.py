#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#UserID::MovieID::Rating::Timestamp
df_user=pd.read_csv("users.dat",sep='::',names=["UserID","Gender","Age","Occupation","Zip-code"],engine="python")


# In[3]:


df_user


# In[4]:


df_movies=pd.read_csv("movies.dat", sep='::',names=["MovieID","Title","Genres"],engine="python")


# In[5]:


df_movies


# In[6]:


df_ratings=pd.read_csv("ratings.dat", sep='::',names=["UserID","MovieID","Rating","Timestamp"],engine="python")


# In[7]:


df_ratings


# In[8]:


df_movies.shape


# In[9]:


df_ratings.shape


# In[10]:


df_user.shape


# #### Create a new dataset [Master_Data] with the following columns MovieID Title UserID Age Gender Occupation Rating.

# In[11]:


dfMovieRatings=df_movies.merge(df_ratings,on="MovieID",how="inner")


# In[12]:


dfMovieRatings


# In[13]:


dfMaster=dfMovieRatings.merge(df_user,on="UserID",how="inner")


# In[14]:


dfMaster


# In[15]:


dfMaster.isnull().sum().any()


# In[16]:


dfMaster.to_csv("Master data 1. csv")


# ##   Explore the datasets using visual representations (graphs or tables), also include your comments on the following:
# #### 1.User Age Distribution
# #### 2.User rating of the movie “Toy Story”
# #### 3.Top 25 movies by viewership rating
# #### 4.Find the ratings for all the movies reviewed by for a particular user of user id = 2696

# In[17]:


#User Age Distribution
dfMaster["Age"].value_counts()


# In[18]:



dfMaster['Age'].value_counts().plot(kind='bar')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('User Age Distribution')
plt.show()


# In[19]:


sns.countplot(x=dfMaster["Age"])


# In[20]:


#User rating of the movie “Toy Story"


# In[21]:


dfMaster.head()


# In[22]:


toystory=dfMaster[dfMaster["Title"].str.contains("Toy Story")==True]
toystory


# In[23]:


toystory.groupby(["Title","Rating"]).size()


# In[24]:


toystory.groupby(["Title","Rating"]).size().plot(kind="barh")


# In[25]:


toystory.groupby(["Title","Rating"]).size().unstack().plot(kind="barh",legend=True)


# In[26]:


# Top 25 movies by viewership rating
dfTop25=dfMaster.groupby(["Title"]).size().sort_values(ascending=False)[:25]


# In[27]:


dfTop25


# In[28]:


plt.figure(figsize=(10,7))
dfTop25=dfMaster.groupby(["Title"]).size().sort_values(ascending=False)[:25].plot(kind="barh")


# In[29]:


# Find the ratings for all the movies reviewed by for a particular user of user id = 2696


# In[30]:


dfMaster[dfMaster["UserID"]==2696].shape[0]


# ### Feature Engineering:
# #### Use column genres:
# 
# #### 1. Find out all the unique genres (Hint: split the data in column genre making a list and then process the data to find out only the unique categories of genres)
# #### 2.Create a separate column for each genre category with a one-hot encoding ( 1 and 0) whether or not the movie belongs to that genre. 
# #### 3.Determine the features affecting the ratings of any particular movie.
# #### 4.Develop an appropriate model to predict the movie ratings

# In[31]:


# Find out all the unique genres (Hint: split the data in column genre making a list and then process the data to find out only the unique categories of genres


# In[32]:


dfMaster["Genres"]


# In[33]:


dfGenres=dfMaster["Genres"].str.split("|")


# In[34]:


dfGenres


# In[35]:


listgenres=set()
for genre in dfGenres:
    listgenres=listgenres.union(set(genre))
    


# In[36]:


listgenres


# In[37]:


len(listgenres)


# In[38]:


#Create a separate column for each genre category with a one-hot encoding ( 1 and 0) whether or not the movie belongs to that genre.


# In[39]:


GenresOneHot=dfMaster["Genres"].str.get_dummies("|")


# In[40]:


GenresOneHot


# In[41]:


dfMaster=pd.concat([dfMaster,GenresOneHot],axis=1)


# In[42]:


dfMaster


# In[43]:


dfMaster.to_csv("Master Copy2.csv")


# In[44]:


#Determine the features affecting the ratings of any particular movie.¶


# In[45]:


dfMaster.columns


# In[46]:


# Rating VS Gender


# In[64]:


dfMaster["Gender"]


# In[48]:


# Convertinng categorical data into numbers


# In[49]:


dfMaster1=dfMaster.copy()


# In[59]:


dfMaster1["Gender"]=dfMaster1["Gender"].replace("M","0")


# In[60]:


dfMaster1["Gender"]=dfMaster1["Gender"].replace("F","1")


# In[61]:


dfMaster1["Gender"]


# In[63]:


dfMaster1.info()


# In[65]:


dfMaster1["Gender"]=dfMaster1["Gender"].astype(int)


# In[67]:


dfMaster1.dtypes


# In[68]:


# Find relationship between gender and rating


# In[71]:


GenderAffecting=dfMaster1.groupby(["Gender","Rating"]).size()


# In[72]:


GenderAffecting


# In[74]:


GenderAffecting=dfMaster1.groupby(["Gender","Rating"]).size().unstack().plot(kind="barh",legend=True)


# ### Gender is affecting the rating

# In[83]:



AgeAffecting=dfMaster1.groupby(["Age","Rating"]).size().unstack().plot(kind="bar",legend=True,figsize=(10,7))


# ### Age is affecting the rating

# In[85]:


# Replationship btn occupaton vs rating


# In[91]:


AgeAffecting=dfMaster1.groupby(["Occupation","Rating"]).size().unstack().plot(kind="bar",legend=True,figsize=(10,9))


# ### Occupation id affecting the rating

# In[95]:


# Develop an appropriate model to predict the movie ratings


# In[96]:


#Prepare i/p data


# In[99]:


dfMaster1.shape


# In[100]:


new_data=dfMaster1[:500]


# In[101]:


new_data


# In[102]:


new_data.columns


# In[106]:


X=new_data[["MovieID","Age","Occupation","Gender"]].values


# In[107]:


X


# In[108]:


Y=new_data[["Rating"]].values


# In[109]:


Y


# In[110]:


# Create train data and test data


# In[111]:


from sklearn.model_selection import train_test_split


# In[112]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=42)


# In[115]:


print(X_train.shape)


# In[118]:


print(X_test.shape)


# In[119]:


print(Y_train.shape)
print(Y_test.shape)


# In[120]:


# Applying Machine Learning Algo


# In[121]:


from sklearn.linear_model import LinearRegression


# In[122]:


lr=LinearRegression()


# In[123]:


lr.fit(X_train,Y_train) #Application of LR on training data


# In[124]:


Y_predict=lr.predict(X_test)


# In[125]:


Y_predict


# In[126]:


Y_test


# In[131]:


#Print the error
from sklearn.metrics import mean_squared_error
print('mean squared error',mean_squared_error(Y_test,Y_predict))


# In[ ]:




