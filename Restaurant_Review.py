# -*- coding: utf-8 -*-
"""

@author: Ajit
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []

for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', data['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english')) or word == 'not']
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = data.iloc[:,1:2].values

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X, y)

y_pred = classifier.predict(X)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)
count = [1 if y[i] == y_pred[i] else 0 for i in range(0,1000) ]
accuracy = count.count(1)/1000 * 100


# type your review in the review below  to test
review = "food is very bad not coming again"
review = re.sub('[^a-zA-Z]', ' ', review)
review = review.lower()
review = review.split()
review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
review = ' '.join(review)
review = [review]
review = cv.transform(review).toarray()
predict = classifier.predict(review)

if predict == 0:
    print("Negative Review")
else:
    print("Positive Review")
