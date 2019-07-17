# --------------
#Importing header files

import pandas as pd

import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split


# Code starts here
data=pd.read_csv(path)
y=data['paid.back.loan']
X=data.drop(['customer.id','paid.back.loan'],axis=1)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
# Code ends here


# --------------
#Importing header files
import matplotlib.pyplot as plt

# Code starts here

fully_paid=y_train.value_counts()
#plt.bar(fully_paid)
plt.show()
print(fully_paid)
# Code ends here


# --------------
#Importing header files
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Code starts here

X_train['int.rate']=X_train['int.rate'].str.replace('%', '')
X_train['int.rate'] = X_train['int.rate'].astype(float)
X_train['int.rate']=X_train['int.rate']/100


X_test['int.rate']=X_test['int.rate'].str.replace('%', '')
X_test['int.rate'] = X_test['int.rate'].astype(float)
X_test['int.rate']=X_test['int.rate']/100

num_df=X_train.select_dtypes(include=[int,float])
cat_df=X_train.select_dtypes(exclude=[int,float])

# Code ends here


# --------------
#Importing header files
import seaborn as sns


# Code starts here
cols=num_df.columns
fig,axes=plt.subplots(nrows=9,ncols=1)
for i in cols:
    #axes=fig.add_subplots(nrows,i+1)
    #ax = sns.boxplot(x=y_train, y=num_df[cols[i]] ,ax=axes[i])
    plt.show()

# Code ends here


# --------------
# Code starts here

cols=cat_df.columns
fig,axes=plt.subplots(2,2, figsize=(10,20))

#Looping across rows
for i in range(2): 
    for j in range(2):        
    
    #Plotting boxplot
        sns.countplot(x=X_train[cols[i*2+j]], hue=y_train , ax=axes[i,j])
    #Avoiding subplots overlapping
fig.tight_layout()    


# Code ends here


# --------------
#Importing header files
from sklearn.tree import DecisionTreeClassifier

# Code starts here
#for col in cat_df.columns:
#X_train[cat_df.columns]=X_train[cat_df.columns].fillna('NA')
#for col in cat_df.columns:
#X_test[cat_df.columns]=X_test[cat_df.columns].fillna('NA')
for col in cat_df.columns:
    X_train[col].replace('NaN','NA',inplace=True)

    X_test[col].replace('NaN','NA',inplace=True)

    le=LabelEncoder()

    X_train[col]=le.fit_transform(X_train[col])


    X_test[col]=le.transform(X_test[col])
#le=LabelEncoder()
#X_train['purpose']=le.fit_transform(X_train['purpose'])
#for col in cat_df.columns:
    #X_train[col]=le.fit_transform(X_train[col])

#X_test['purpose']=le.transform(X_test['purpose'])
#for col in cat_df.columns:
    #X_test[col]=le.transform(X_test[col])

y_train=y_train.str.replace('No','0').replace('Yes','1').astype('int')
y_test=y_test.str.replace('No','0').replace('Yes','1').astype('int')
model=DecisionTreeClassifier(random_state=0)
model.fit(X_train,y_train)
acc=model.score(X_test,y_test)


# Code ends here


# --------------
#Importing header files
from sklearn.model_selection import GridSearchCV

#Parameter grid
parameter_grid = {'max_depth': np.arange(3,10), 'min_samples_leaf': range(10,50,10)}

# Code starts here

model_2=DecisionTreeClassifier(random_state=0)
p_tree=GridSearchCV(estimator=model_2, param_grid=parameter_grid , cv=5)
p_tree.fit(X_train,y_train)
acc_2=p_tree.score(X_test,y_test)
# Code ends here


# --------------
#Importing header files

from io import StringIO
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn import metrics
from IPython.display import Image
import pydotplus

# Code starts here
dot_data=export_graphviz(decision_tree=p_tree.best_estimator_, out_file=None, feature_names=X.columns, filled = True , class_names=['loan_paid_back_yes','loan_paid_back_no'])

graph_big=pydotplus.graph_from_dot_data(dot_data)
# show graph - do not delete/modify the code below this line
img_path = user_data_dir+'/file.png'
graph_big.write_png(img_path)

plt.figure(figsize=(20,15))
plt.imshow(plt.imread(img_path))
plt.axis('off')
plt.show() 

# Code ends here


