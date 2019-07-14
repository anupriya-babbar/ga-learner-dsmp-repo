# --------------
import pandas as pd
from sklearn import preprocessing

#path : File path

# Code starts here
dataset=pd.read_csv(path)

# read the dataset



# look at the first five columns

dataset.head()
# Check if there's any column which is not useful and remove it like the column id

dataset.drop('Id',axis=1,inplace=True)
# check the statistical description
dataset.describe()


# --------------
# We will visualize all the attributes using Violin Plot - a combination of box and density plots
import seaborn as sns
from matplotlib import pyplot as plt

#names of all the attributes 

cols=list(dataset.columns)
#number of attributes (exclude target)
size=len(cols)-1


#x-axis has target attribute to distinguish between classes

x=dataset.iloc[:,-1]
y=dataset.iloc[:,:-1]
#y-axis shows values of an attribute


#Plot violin for all attributes
#fig, ax = pyplot.subplots(figsize =(9, 7)) 

for feat in cols :
    ax=sns.violinplot( x = dataset[feat])
    plt.show()


# --------------
import numpy
#threshold=0.75
upper_threshold = 0.5
lower_threshold=-0.5

# no. of features considered after ignoring categorical variables
num_features = 10
# create a subset of dataframe with only 'num_features'
subset_train=dataset.iloc[:,0:10]
data_corr=subset_train.corr()
sns.heatmap(data_corr)
correlation=data_corr.unstack().sort_values(kind='quicksort')
corr_var_list=correlation[((correlation>upper_threshold) & (correlation!=1)|(correlation< lower_threshold)
&(correlation!=1))]
#Print correlations and column names


# --------------
#Import libraries 
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler

# Identify the unnecessary columns and remove it 
dataset.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True)

X=dataset

Y=X[['Cover_Type']].copy()
X.drop("Cover_Type",axis=1,inplace=True)
# Scales are not the same for all variables. Hence, rescaling and standardization may be necessary for some algorithm to be applied on it.
#cross_validation.KFold(10, n_folds=2, shuffle=True, random_state=1):

X_train,X_test,Y_train,Y_test= cross_validation.train_test_split(X, Y, test_size=0.2,random_state=0)
#Standardized
#Apply transform only for non-categorical data

scaler=StandardScaler()
X_train_con=X_train.select_dtypes(include=['int32','int64','float'])
X_test_con=X_test.select_dtypes(include=['int32','int64','float'])
list_con=list(X_train_con.columns)
for feat in list_con:
    X_train_con[feat]=scaler.fit_transform(X_train_con[[feat]])
X_train_temp=X_train_con
for feat in list_con:
    X_test_con[feat]=scaler.fit_transform(X_test[[feat]])
X_test_temp=X_test_con
X_train_cat=X_train.select_dtypes(exclude=['int32','int64','float'])
X_test_cat=X_test.select_dtypes(exclude=['int32','int64','float'])
scaled_features_train_df=pd.concat([X_train_temp,X_train_cat],1)
scaled_features_test_df=pd.concat([X_test_temp,X_test_cat],1)
print(scaled_features_train_df)

#Concatenate non-categorical data and categorical



# --------------
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif


# Write your solution here:

skb=SelectPercentile(score_func=f_classif,percentile=20)

predictors=skb.fit_transform(X_train1,Y_train)
scores=skb.scores_.tolist()
#print(X_train1)
Features=X.columns
dataframe=pd.DataFrame({'Features':Features,'scores':scores})
dataframe=dataframe.sort_values(by=['scores'],ascending=False)
top_k_predictors=dataframe[dataframe.scores>=dataframe.scores.quantile(.80)].Features.tolist()
print(dataframe.scores.quantile(0.80))
#print(np.percentile(dataframe.scores,80))
print(top_k_predictors)
#x_new = selector.transform(X) # not needed to get the score
#scores = selector.scores_


# --------------
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
clf=OneVsRestClassifier(LogisticRegression())
clf1=OneVsRestClassifier(LogisticRegression())
#model_fit_all_features=LogisticRegression()

#model_fit_all_features=clf1.fit(X_train,Y_train)
#prediction_all_feature=model_fit_all_features.predict(X_test)
#score_all_features=accuracy_score(Y_test,prediction_all_feature)
#model_fit_top_features=clf.fit(scaled_features_train_df,top_k_predictors)
#XX=scaled_features_train_df[top_k_predictors]
#model_fit_top_features.fit(XX,Y_train)
#XX_test=scaled_features_test_df[top_k_predictors]
#y_pred=model_fit_top_features.predict(X_test)
#predictions_top_features=model_fit_top_features.predict(scaled_features_test_df)
#score_top_features=accuracy_score(Y_test,predictions_top_features)

clf = LogisticRegression(random_state=0, multi_class='ovr')
clf1 = LogisticRegression(random_state=0, multi_class='ovr')

model_fit_all_features = clf1.fit(X_train, Y_train)
predictions_all_features = clf1.predict(X_test)

score_all_features = accuracy_score(predictions_all_features, Y_test)

print('score_all_features', score_all_features)
# print(scaled_features_train_df.head())
# print(scaled_features_train_df[top_k_predictors].shape)

model_fit_top_features = clf.fit(scaled_features_train_df[top_k_predictors], Y_train)
predictions_top_features = clf.predict(scaled_features_test_df[top_k_predictors])
score_top_features = accuracy_score(predictions_top_features, Y_test)

print('score_top_features',score_top_features)


