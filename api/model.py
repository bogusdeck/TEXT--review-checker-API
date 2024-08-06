import pickle 
import pandas as pd  
import numpy as np 
# import pywedge as pw
import sklearn as sk
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split

df = pd.read_csv('deceptive-opinion.csv')

df1 = df[['deceptive','text']]

df1.loc[df1['deceptive']=='deceptive','deceptive'] = 0
df1.loc[df1['deceptive']=='truthful', 'deceptive'] = 1

X = df1['text']
Y = np.asarray(df1['deceptive'], dtype =int)

X_train,X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3, random_state=109)

nb = MultinomialNB()

cv = CountVectorizer()
x = cv.fit_transform(X_train)
y = cv.transform(X_test)


nb.fit(x, y_train)
pickle.dump(nb,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print(model)
nb.predict(y)

nb.score(x, y_train)

nb.score(y,y_test)

clf = svm.SVC(kernel='linear') 

clf.fit(x,y_train)

y_pred=clf.predict(y)

y_pred

clf.score(x,y_train)

hehe = clf.score(y,y_test)
print(hehe)


