#!/usr/bin/env python
# coding: utf-8

# In[118]:


import pandas as pd
import numpy as np
import tarfile
import os


# In[119]:


cd


# In[120]:


tar = tarfile.open(r"C:\Users\Rutvik\Downloads\enron1.tar.gz")
tar.getmembers()


# In[121]:


#tar2 = tarfile.open(r"C:\Users\Rutvik\Downloads\GP.tar.gz")
#tar2.getmembers()


# In[122]:


# data = pd.read_csv(r"C:\Users\Rutvik\Downloads\spambase\spambase.data",)
# data


# In[123]:


#ham_file_locations = os.listdir(str(tar.[1]))


# In[124]:


t = tar.list()


# In[125]:


tar = tarfile.open(r"C:\Users\Rutvik\Downloads\enron1.tar.gz")
tar.extractall(path=".\Datasets")
tar.close()


# In[126]:


tar = tarfile.open(r"C:\Users\Rutvik\Downloads\enron2.tar.gz")
tar.extractall()
tar.close()


# In[127]:


ham_file_locations = os.listdir("\Datasets\enron1\ham")
spam_file_locations = os.listdir("\Datasets\enron1\spam")


# In[128]:


ham_file_locations


# In[129]:


# def load_data():
#     print("Loading the data...")
#     ham_file_locations = os.listdir("/Datasets/enron1/ham")
#     spam_file_locations = os.listdir("/Datasets/enron1/spam")
#     data = []
#     #data = []
    
#     for file in ham_file_locations:
#         f = open("Datasets/enron1/ham/"+ file, encoding="utf-8", mode="r")
#         text = str(f.read())
#         data.append([text, "ham"])
#         f.close()
        
#     for file in spam_file_locations:
#         f = open("Datasets/enron1/spam/" + file, encoding="utf-8", mode="r", errors="ignore")
#         text = str(f.read())
#         data.append([text, "spam"])
#         f.close()
    
#     data = np.array(data)
#     print("Data Loaded.")
#     return data


# In[130]:


#load_data()


# ## Loading the dataset into a dataframe

# In[131]:


pathwalk = os.walk(r"/Datasets/enron1")


# In[132]:


data_ham = []
data_spam = []


# In[133]:


for root, dir, file in pathwalk:
    if 'ham' in str(file):
        for obj in file:
            with open(root + "/" + obj, encoding='latin1') as ip:
                data_ham.append(" ".join(ip.readlines()))

    elif 'spam' in str(file):
        for obj in file:
            with open(root + "/" + obj, encoding='latin1') as ip:
                data_spam.append(" ".join(ip.readlines()))


# In[134]:


data_ham


# In[135]:


new_data_ham = list(set(data_ham))
new_data_spam = list(set(data_spam))


# In[136]:


len(new_data_spam)


# In[137]:


len(new_data_ham)


# In[138]:


emails = new_data_ham + new_data_spam
labels = [0]*len(new_data_ham) + [1]*len(new_data_spam)

df = pd.DataFrame({"emails": emails, "label": labels})


# In[139]:


df


# In[140]:


df.label.hist(bins=3)


# In[141]:


df.label.value_counts()


# ## Pre Processing the data

# In[142]:


import string
import re


# In[143]:


string.punctuation


# In[144]:


# Removing the punctuation 
def remove_punctuation(text):
    punctuation_free = "".join([i for i in text if i not in string.punctuation])
    return punctuation_free

df["cleaned"] = df.emails.apply(lambda x: remove_punctuation(x))
df.head()


# In[145]:


df["cleaned"] = df.cleaned.apply(lambda x: x.lower())
df.head()


# In[146]:


def apply_cleaning(text):
    #text = text.lower()
    #text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('\n', '', text)
    return text

cleaning_1 = lambda x: apply_cleaning(x)
df_clean = pd.DataFrame(df.emails.apply(cleaning_1))
df_clean.head()


# In[147]:


# def apply_cleaning_2(text):
#     text = re.sub('\n', '', text)
#     return text

# cleaning_2 = lambda x: apply_cleaning_2(x)
# df_cleaner = df_clean.emails.apply(cleaning_2)
# df_cleaner.head()


# In[148]:


import nltk


# In[149]:


#nltk.download()


# In[150]:


#df_clean['emails'] = df_cleaner
#df_clean.head()


# In[151]:


import time
line = df_clean["emails"][0]

def method_1(line):
    line = re.sub(r'[^\w\s]', '', line)
    return line
def method_2(line):
    line = re.sub('[%s]' % re.escape(string.punctuation), '', line)
    return line

begin = time.time()
line_pro1 = method_1(line)
time.sleep(1)
end = time.time()
print("Time taken by method 1:", end-begin)

begin = time.time()
line_pro2 = method_2(line)
time.sleep(1)
end = time.time()
print("Time taken by method 2:", end-begin)


# In[152]:


from collections import Counter
t = nltk.word_tokenize(df_clean.emails[0])
c = Counter(t)
print(-int(len(c)*0.1))
print(len(c))
uncommon = c.most_common()[:-int(len(c)*0.1):-1]
print(uncommon)
print()
uncommon2 = c.most_common()[:-1]
print(uncommon2)


# In[153]:


def broom_preprocess(text):
    text = apply_cleaning(text)
    tokens = nltk.word_tokenize(text)
    
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w, pos='a') for w in tokens]
    tokens = ' '.join(tokens)
    
    return tokens


# In[154]:


df_processed = pd.DataFrame()
df_processed["emails"] = [broom_preprocess(email) for email in df.emails]
df_processed.head()


# In[155]:


df_processed["labels"] = df.label
df_processed.head()


# In[156]:


# df_clean = pd.DataFrame()
# df_clean["emails"] = [apply_cleaning(email) for email in df.emails]
# df_clean.head()
# df_clean["labels"] = df.label
# df_clean.head()


# In[157]:


corpus = []
stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.WordNetLemmatizer()

for i in range(0, df.shape[0]):
    email = re.sub(pattern='[^a-zA-Z]', repl=' ', string=df.emails[i])
    email = email.lower()
    words = email.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(w, pos='a') for w in words]
    email = ' '.join(words)
    corpus.append(email)


# In[158]:


corpus


# In[159]:


#!pip install imbalanced-learn


# ## Train Test Split

# In[160]:


from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


# In[161]:


X = df_processed["emails"]
y = df_processed["labels"]


# In[162]:


df = df.drop(["cleaned"], axis=1)
df.head()


# In[163]:


df.to_csv('enron_spam_2df.csv')


# In[164]:


print(Counter(y))


# In[165]:


np.random.seed(42)


# In[166]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y)


# In[167]:


print("X train:", X_train.shape)
print("X test:", X_test.shape)
print("y train:", y_train.shape)
print("y test:", y_test.shape)


# In[168]:


print(Counter(y_train))


# In[169]:


print(Counter(y_test))


# In[170]:


X_train


# ## Converting words to numbers

# In[171]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline


# In[172]:


tfidf = TfidfVectorizer(lowercase=True)
X_train_tfidf = tfidf.fit_transform(X_train)


# ## Model Selection and Testing

# In[173]:


from sklearn.naive_bayes import MultinomialNB
nbc = MultinomialNB()
model_1 = nbc.fit(X_train_tfidf, y_train)


# In[174]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


# In[175]:


scores_tfidf = cross_val_score(model_1, X_train_tfidf, y_train, cv=5, scoring='accuracy')
print("Validation accuracy:", np.mean(scores_tfidf))


# In[176]:


X_test_tfidf = tfidf.transform(X_test)
preds = model_1.predict(X_test_tfidf)
res_1 = accuracy_score(y_test, preds)
print("Test accuracy:", res_1)


# In[177]:


cv = CountVectorizer(lowercase=True, stop_words='english')
X_train_cv = cv.fit_transform(X_train)


# In[178]:


pipe_NB = Pipeline([('cv', CountVectorizer(lowercase=True, stop_words='english')), 
                   ('naive_bayes', MultinomialNB())])
model_2 = pipe_NB.fit(X_train, y_train)
scores_cv = cross_val_score(model_2, X_train, y_train, cv=5, scoring='accuracy')
print("Validation accuracy:", np.mean(scores_cv))


# In[179]:


X_test_cv = cv.transform(X_test)
preds_2 = model_2.predict(X_test)
res_2 = accuracy_score(y_test, preds_2)
print("Test accuracy:", res_2)


# In[180]:


from sklearn.metrics import confusion_matrix
cnf_mat = confusion_matrix(y_test, preds_2)
print("   T      N")
print("T", cnf_mat[0])
print("F", cnf_mat[1])


# In[181]:


from sklearn.metrics import recall_score, precision_score, f1_score
print("Recall Score:", recall_score(y_test, preds_2))
print("Precision Score:", precision_score(y_test, preds_2))
print("F1 Score:", f1_score(y_test, preds_2))


# In[182]:


import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
precision, recall, threshold = precision_recall_curve(y_test, preds_2)
plt.plot(recall, precision, marker='.', label="Naive Bayes Spam Classifier")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()


# ## Trying out the Logistic Regressor

# In[183]:


from sklearn.linear_model import LogisticRegression


# In[184]:


lr = LogisticRegression(solver="lbfgs", random_state=42)
model_3 = lr.fit(X_train_cv, y_train)


# In[185]:


scores_cv_lr = cross_val_score(model_3, X_train_cv, y_train, cv=5, scoring='accuracy')
print("Validation accuracy:", np.mean(scores_cv_lr))


# In[186]:


preds_3 = model_3.predict(X_test_cv)
res_3 = accuracy_score(y_test, preds_3)
print("Test accuracy:", res_3)


# In[187]:


cnf_mat_2 = confusion_matrix(y_test, preds_3)
print("   T      N")
print("T", cnf_mat_2[0])
print("F", cnf_mat_2[1])


# In[188]:


print("Recall Score:", recall_score(y_test, preds_3))
print("Precision Score:", precision_score(y_test, preds_3))
print("F1 Score:", f1_score(y_test, preds_3))


# In[189]:


precision, recall, threshold = precision_recall_curve(y_test, preds_3)
plt.plot(recall, precision, marker='.', label="Logistic Regressor")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()


# In[190]:


# Logistic Regressor is slightly better


# In[191]:


lr_2 = LogisticRegression(solver="liblinear", random_state=42)
#model_4 = lr_2.fit(X_train_cv, y_train)
pipeline = Pipeline([('cv', CountVectorizer(tokenizer=broom_preprocess, lowercase=True)),
                    ('log_reg', LogisticRegression(solver="liblinear", random_state=42))])
model_4 = pipeline.fit(X_train, y_train)


# In[192]:


scores_cv_lr_2 = cross_val_score(model_4, X_train, y_train, cv=5, scoring='accuracy')
print("Validation accuracy:", np.mean(scores_cv_lr_2))
preds_4 = model_4.predict(X_test)
res_4 = accuracy_score(y_test, preds_4)
print("Test accuracy:", res_4)


# In[193]:


cnf_mat_3 = confusion_matrix(y_test, preds_4)
print("   T      N")
print("T", cnf_mat_3[0])
print("F", cnf_mat_3[1])


# In[194]:


print("Recall Score:", recall_score(y_test, preds_4))
print("Precision Score:", precision_score(y_test, preds_4))
print("F1 Score:", f1_score(y_test, preds_4))


# In[195]:


precision, recall, threshold = precision_recall_curve(y_test, preds_4)
plt.plot(recall, precision, marker='.', label="Logistic Regressor")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()


# In[196]:


import joblib

joblib.dump(model_2, "pipe_NB.pkl")


# In[ ]:




