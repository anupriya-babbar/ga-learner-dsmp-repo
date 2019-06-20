# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data=pd.read_csv(path)
#data['Rating']=data['Rating'].astype(int)
#data['Rating']=data['Rating'].dropna()
#data['Rating']=data['Rating'].astype(int)
sns.distplot(data['Rating'].dropna())
data=data[data['Rating']<=5]
sns.distplot(data['Rating'])
plt.show()



# --------------
# code starts here


# code ends here
total_null=data.isnull().sum()
percent_null=total_null/data.isnull().count()

missing_data=pd.concat([total_null,percent_null],1,keys=['Total','Percent'])
print(missing_data)
data.dropna(inplace=True)
total_null_1=data.isnull().sum()
percent_null_1=total_null_1/data.isnull().count()
missing_data_1=pd.concat([total_null_1,percent_null_1],1,keys=['Total','Percent'])
print(missing_data_1)


# --------------

#Code starts here
g = sns.catplot(x="Category", y="Rating", data=data,kind='box',height=10)
g.set_xticklabels(rotation=90)
g.set_titles("Rating vs Category [BoxPlot]")


#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
install=data['Installs'].copy()
install.value_counts()

data['Installs']=data['Installs'].str.replace(',','')
data['Installs']=data['Installs'].str.replace('+','')
data['Installs'] = data['Installs'].astype(int)
le=LabelEncoder()
le.fit(data['Installs'])
data['Installs']=le.transform(data['Installs'])
r=sns.regplot( x="Installs", y="Rating",data=data)
r.set_title("Rating vs Installs [RegPlot]")




#Code ends here



# --------------
#Code starts here
price=data['Price'].copy()
price.value_counts()
data['Price']=data['Price'].str.replace("$","")
data['Price']=data['Price'].astype(float)
r=sns.regplot(x="Price", y="Rating",data=data)
r.set_title("Rating vs Price [RegPlot]")



#Code ends here


# --------------

#Code starts here
print(data['Genres'].unique())

data['Genres']=data['Genres'].str.split(';').str[0]
gr_mean=data.groupby("Genres",as_index=False)[['Genres','Rating']].mean()
print(gr_mean.describe())
gr_mean=gr_mean.sort_values(by='Rating')

print(gr_mean)
#Code ends here


# --------------

#Code starts here

data['Last Updated'] = pd.to_datetime(data['Last Updated'])
max_date=max(data['Last Updated'])
data['Last Updated Days']=max_date-data['Last Updated']
data['Last Updated Days']=data['Last Updated Days'].dt.days
r=sns.regplot(x='Last Updated Days',y='Rating', data=data)
r.set_title("Rating vs Last Updated [RegPlot]")
#Code ends here


