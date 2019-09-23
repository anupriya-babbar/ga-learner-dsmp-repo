# --------------
import pandas as pd
from collections import Counter

# Load dataset
data=pd.read_csv(path)
data.isnull().sum()
data.describe()


# --------------
import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style(style='darkgrid')

# Store the label values 

label=data['Activity']
sns.countplot(x=label)
# plot the countplot



# --------------
# make the copy of dataset

data_copy=data.copy()
data_copy['duration']=""

duration_df=(data_copy.groupby([label[(label=='WALKING_UPSTAIRS')|(label=='WALKING_DOWNSTAIRS')], 'subject'])['duration'].count()*1.28)
duration_df=pd.DataFrame(duration_df)
#print(duration_df)
#data_copy['duration']=duration_df
plot_data=duration_df.reset_index().sort_values('duration',ascending=False)
plot_data['Activity']=plot_data['Activity'].map({'WALKING_UPSTAIRS':'Upstairs','WALKING_DOWNSTAIRS':'Downstairs'})
plt.figure(figsize=(15,5))
sns.barplot(data=plot_data,x='subject',y='duration',hue='Activity')

# Create an empty column 
plt.title("partcipation compared")
plt.xlabel("participants")
plt.ylabel("Duration")
plt.show()


# Calculate the duration




# Sort the values of duration

sns.barplot(data=plot_data, x='subject', y='duration', hue='Activity')



# --------------
#exclude the Activity column and the subject column
feature_col=data.drop(['Activity','subject'],1)
#feature_cols.head()

feature_cols=feature_col.columns
#Calculate the correlation values
correlated_values=feature_col.corr()
#print(correlated_values)
#stack the data and convert to a dataframe

correlated_values = correlated_values.stack().to_frame().reset_index().rename(columns={'level_0': 'Feature_1', 'level_1': 'Feature_2', 0:'Correlation_score'})
#correlated_values.head()

#create an abs_correlation column

correlated_values['abs_correlation']=abs(correlated_values['Correlation_score'])

#corr_var_list=correlated_values['abs_correlation']
s_corr_list=correlated_values.sort_values('abs_correlation',ascending=False)
#Picking most correlated features without having self correlated pairs
top_corr_fields=s_corr_list[(s_corr_list['abs_correlation']>.8) & (s_corr_list['Feature_1']!=s_corr_list['Feature_2'])]
print(top_corr_fields)



# --------------
# importing neccessary libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support as error_metric
from sklearn.metrics import confusion_matrix, accuracy_score

# Encoding the target variable

le=LabelEncoder()

# split the dataset into train and test
y=data['Activity']
y=le.fit_transform(y)
X=data.drop("Activity",1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=40)
classifier=SVC()
clf=classifier.fit(X_train,y_train)
y_pred=clf.predict(X_test)
precision, accuracy, f_score, support =error_metric(y_test, y_pred, average='weighted')
model1_score=clf.score(X_test,y_test)

# Baseline model 




# --------------
# importing libraries
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import recall_score,f1_score,precision_score,accuracy_score

lsvc=LinearSVC(C=0.01,dual=False,random_state=42,penalty='l1')
lsvc.fit(X_train,y_train)
model_2=SelectFromModel(estimator=lsvc,prefit=True)
new_train_features=model_2.transform(X_train)
new_test_features=model_2.transform(X_test)
# Feature selection using Linear SVC
classfier_2=SVC()
clf_2=classfier_2.fit(new_train_features,y_train)
y_pred_new=clf_2.predict(new_test_features)
model2_score=accuracy_score(y_test,y_pred_new)

f_score=f1_score(y_test,y_pred_new,average='weighted')
precision=precision_score(y_test,y_pred_new,average='weighted')
recall=recall_score(y_test,y_pred_new,average='weighted')
# model building on reduced set of features




# --------------
# Importing Libraries
from sklearn.model_selection import GridSearchCV

# Set the hyperparmeters
parameters = {'kernel':['linear', 'rbf'], 'C':[100,20,1,0.1]}
#svc = svm.SVC(gamma="scale")
selector= GridSearchCV(SVC(),param_grid= parameters, scoring='accuracy')


# Usage of grid search to select the best hyperparmeters

selector.fit(new_train_features,y_train)
print(selector.best_params_)
means=selector.cv_results_['mean_test_score']
stds=selector.cv_results_['std_test_score']
params=selector.cv_results_['params']

classifier_3=SVC(C=100,kernel='rbf')
clf_3=classifier_3.fit(new_train_features,y_train)
y_pred_final=clf_3.predict(new_test_features)
model3_score=accuracy_score(y_test,y_pred_final)
precision,recall,f_score,support=error_metric(y_test,y_pred_final,average='weighted')

# Model building after Hyperparameter tuning





