#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
from tqdm import tqdm
from nltk import word_tokenize
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
import re
import seaborn as sns
import matplotlib.pyplot as plt
from collections import  Counter
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.stem import PorterStemmer
from spacy.lang.en import English
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


os.getcwd()


# In[3]:


data = pd.read_csv("fake_job_postings.csv", sep = ",")


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


print(data.columns)
data.describe()


# In[7]:


columns=['job_id', 'telecommuting', 'has_company_logo', 'has_questions', 'salary_range', 'employment_type', 'location']
for col in columns:
    del data[col]

data.fillna(' ', inplace=True)


# In[8]:


data.head()


# In[9]:


sns.countplot(data.fraudulent)
data.groupby('fraudulent').count()['title'].reset_index().sort_values(by='title',ascending=False)


# In[10]:


education = dict(data.required_education.value_counts())
del education[' ']
plt.figure(figsize=(15, 8))
plt.bar(education.keys(), education.values())
plt.xlabel('no. of jobs', size=10)
plt.ylabel('Education')
plt.xticks(rotation=80)
plt.show()


# In[11]:


experience = dict(data.required_experience.value_counts())
del experience[' ']
plt.figure(figsize=(15, 8))
plt.bar(experience.keys(), experience.values())
plt.xlabel('Experience', size=10)
plt.ylabel('no. of jobs', size=10)
plt.xticks(rotation=80)
plt.show()


# In[12]:


print(data.title.value_counts()[:10])


# In[13]:


print(data.department.value_counts()[:10])


# In[14]:


print(data.industry.value_counts()[:10])


# In[15]:


print(data.function.value_counts()[:10])


# In[16]:


word_count = data['description'].str.split().map(lambda x: len(x))
plt.figure(figsize=(15, 8))
word_count.hist()
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.title('Histogram of Word Count')
plt.show()
word_count.describe()


# In[17]:


word_count = data['company_profile'].str.split().map(lambda x: len(x))
plt.figure(figsize=(15, 8))
word_count.hist()
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.title('Histogram of Word Count')
plt.show()
word_count.describe()


# In[18]:


word_count = data['requirements'].str.split().map(lambda x: len(x))
plt.figure(figsize=(15, 8))
word_count.hist()
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.title('Histogram of Word Count')
plt.show()
word_count.describe()


# In[19]:


word_count = data['benefits'].str.split().map(lambda x: len(x))
plt.figure(figsize=(15, 8))
word_count.hist()
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.title('Histogram of Word Count')
plt.show()
word_count.describe()


# In[20]:


data2 = data.loc[data['fraudulent'] == 1]


# In[21]:


data2.head()


# In[22]:


education = dict(data2.required_education.value_counts())
del education[' ']
plt.figure(figsize=(15, 8))
plt.bar(education.keys(), education.values())
plt.xlabel('no. of jobs', size=10)
plt.ylabel('Education')
plt.xticks(rotation=80)
plt.show()


# In[23]:


experience = dict(data2.required_experience.value_counts())
del experience[' ']
plt.figure(figsize=(15, 8))
plt.bar(experience.keys(), experience.values())
plt.xlabel('Experience', size=10)
plt.ylabel('no. of jobs', size=10)
plt.xticks(rotation=80)
plt.show()


# In[24]:


print(data2.title.value_counts()[:10])


# In[25]:


print(data2.department.value_counts()[:10])


# In[26]:


print(data2.industry.value_counts()[:10])


# In[27]:


print(data2.function.value_counts()[:10])


# In[28]:


word_count = data2['description'].str.split().map(lambda x: len(x))
plt.figure(figsize=(15, 8))
word_count.hist()
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.title('Histogram of Word Count')
plt.show()
word_count.describe()


# In[29]:


word_count = data2['company_profile'].str.split().map(lambda x: len(x))
plt.figure(figsize=(15, 8))
word_count.hist()
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.title('Histogram of Word Count')
plt.show()
word_count.describe()


# In[30]:


word_count = data2['requirements'].str.split().map(lambda x: len(x))
plt.figure(figsize=(15, 8))
word_count.hist()
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.title('Histogram of Word Count')
plt.show()
word_count.describe()


# In[31]:


word_count = data2['benefits'].str.split().map(lambda x: len(x))
plt.figure(figsize=(15, 8))
word_count.hist()
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.title('Histogram of Word Count')
plt.show()
word_count.describe()


# In[32]:


data['text']=data['title']+' '+' '+data['company_profile']+' '+data['description']+' '+data['requirements']+' '+data['benefits']


# In[33]:


data.text[0]


# In[34]:


data.head()


# In[35]:


columns=['title', 'department', 'company_profile', 'description', 'requirements', 'benefits', 'required_experience', 'required_education', 'industry', 'function']
for col in columns:
    del data[col]
    
data.fillna(' ', inplace=True)


# In[36]:


data.head()


# In[37]:


data.head()


# In[38]:


corpus = []
for x in data['text'].str.split():
    corpus.extend(x)


# In[39]:


counter=Counter(corpus)
most=counter.most_common()


# In[40]:


stop=set(stopwords.words('english'))


# In[41]:


most_nostop = []
for word, count in most:
    if word.lower() not in stop:
        most_nostop.append((word, count))


# In[42]:


first_n = 20
x, y= [], []
for word,count in most_nostop[:first_n]:
        x.append(word)
        y.append(count)

plt.figure(figsize=(10, 10))
sns.barplot(x=y,y=x)
print(f"{first_n} najczęściej występujących slów w korpusie")


# In[43]:


def clean(text):
    text=text.lower()
    obj=re.compile(r"<.*?>")                     #removing html tags
    text=obj.sub(r" ",text)
    obj=re.compile(r"https://\S+|http://\S+")    #removing url
    text=obj.sub(r" ",text)
    obj=re.compile(r"[^\w\s]")                   #removing punctuations
    text=obj.sub(r" ",text)
    obj=re.compile(r"\d{1,}")                    #removing digits
    text=obj.sub(r" ",text)
    obj=re.compile(r"_+")                        #removing underscore
    text=obj.sub(r" ",text)
    obj=re.compile(r"\s\w\s")                    #removing single character
    text=obj.sub(r" ",text)
    obj=re.compile(r"\s{2,}")                    #removing multiple spaces
    text=obj.sub(r" ",text)
   
    
    stemmer = PorterStemmer()
    text=[stemmer.stem(word) for word in text.split() if word not in stop]
    
    return " ".join(text)


# In[44]:


tqdm.pandas()
data['text']=data["text"].progress_apply(clean)


# In[45]:


data.head()


# In[46]:


fraudjobs_text = data[data.fraudulent==1].text
actualjobs_text = data[data.fraudulent==0].text


# In[47]:


#fałszywe
STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS
plt.figure(figsize = (16,14))
wc = WordCloud(min_font_size = 3,  max_words = 3000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(str(" ".join(fraudjobs_text)))
plt.imshow(wc,interpolation = 'bilinear')


# In[48]:


#prawdziwe
plt.figure(figsize = (16,14))
wc = WordCloud(min_font_size = 3,  max_words = 3000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(str(" ".join(actualjobs_text)))
plt.imshow(wc,interpolation = 'bilinear')


# In[58]:


X_data = data["text"].copy()
y_data = data["fraudulent"].copy()
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3,random_state=random_state )


# In[60]:


x_train_df = X_train


# In[61]:


x_test_df = X_test


# In[62]:


x_test_df.count()


# In[63]:


y_train.count()


# In[64]:


y_train.value_counts(normalize=True)


# In[65]:


y_test.count()


# In[66]:


y_test.value_counts(normalize=True)


# In[67]:


def getScores(estimator, x, y):
    yPred = estimator.predict(x)
    return (metrics.accuracy_score(y, yPred), 
            metrics.f1_score(y, yPred, average='micro'), 
            metrics.f1_score(y, yPred, average='macro'))

def my_scorer(estimator, x, y):
    a, p, r = getScores(estimator, x, y)
    print("Accuracy: {} | F1 micro: {} | F1 macro: {}".format(a, p, r))
    return a+p+r


# In[68]:


clf = SGDClassifier(loss='hinge', penalty='l2', learning_rate='optimal', alpha=1e-4, random_state=42, class_weight='balanced')
pipe = Pipeline([('tfidf', TfidfVectorizer()),
                ('clf', clf),
               ])


# In[69]:


x_train = x_train_df
scores = cross_val_score(pipe, x_train, np.ravel(y_train), scoring=my_scorer, cv=10)


# In[70]:


pipe.fit(x_train, y_train.values.ravel())


# In[72]:


y_pred = pipe.predict(x_test_df)


# In[73]:


print("SGD Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("SGD F1 micro:",metrics.f1_score(y_test, y_pred, average='micro'))
print("SGD F1 macro:",metrics.f1_score(y_test, y_pred, average='macro'))


# In[74]:


print(classification_report(y_test, y_pred, target_names=['0','1']))


# In[75]:


dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(x_train, y_train)
y_pred = dummy_clf.predict(x_test_df)
print("Dummy classifier Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Dummy classifier F1 micro:",metrics.f1_score(y_test, y_pred, average='micro'))
print("Dummy classifier F1 macro:",metrics.f1_score(y_test, y_pred, average='macro'))


# In[76]:


print(classification_report(y_test, y_pred, target_names=['0','1']))


# In[ ]:




