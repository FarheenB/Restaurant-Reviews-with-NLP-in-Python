#!/usr/bin/env python
# coding: utf-8

# ### Importing the libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix


# ### Importing the dataset

# In[2]:


dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
dataset.head()


# ### Cleaning the texts

# In[3]:


import re
import nltk
#download stopwords
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    #replace chars that are not a-z or A-Z with space
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    #lower case all the chars
    review = review.lower()
    #split the words of the review.Returns a list
    review = review.split()
    #stem the words having variation with the tenses 
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #join the words in the list with space separation
    review = ' '.join(review)
    #all the reviews appended in an array
    corpus.append(review)


# In[4]:


#first review
print(dataset.iloc[0,0:1].values)
#after cleaning the text
print(corpus[0])


# ### Creating the Bag of Words model

# In[5]:


from sklearn.feature_extraction.text import CountVectorizer
#creating sparse matrix
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# In[6]:


#sparse matrix of the words
print(X)


# In[7]:


#likes and dislikes
print(y)


# ### Splitting the dataset into the Training set and Test set

# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# ### Model Report

# In[9]:


def model_report(classifier):
    #fit training set
    classifier.fit(X_train, y_train)
    
    #Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tp, fp, fn, tn=cm.ravel()
    print("\nConfusion Matrix:\n",cm)
    
    #predict accuracy
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    print("\nAccuracy:" ,accuracy*100,"%")
    
    #find precision
    precision = tp / (tp + fp)
    print("\nPrecision:",tp)
    
    #recall
    recall= tp/(tp+fn)
    print("\nRecall:",recall)
    
    #F1 Score
    F1_score = 2 * precision * recall / (precision + recall)
    print("\nF1 Score:",F1_score)
    
    


# ## Training the Naive Bayes model on the Training set

# In[10]:


from sklearn.naive_bayes import GaussianNB
classifier_na = GaussianNB()
print("----Naive Bayers Classifier----")
model_report(classifier_na)


# ## Training the Decision Tree model on the Training set

# In[11]:


from sklearn.tree import DecisionTreeClassifier
classifier_dtc = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
print("----Decision Tree Classifier----")
model_report(classifier_dtc)


# ## Training the Random Forest model on the Training set

# In[12]:


from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
print("----Random Forest Classifier----")
model_report(classifier_rf)

