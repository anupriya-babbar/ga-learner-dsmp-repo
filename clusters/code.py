# --------------
# import packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 



# Load Offers

offers=pd.read_excel(path,sheet_name=0)
# Load Transactions
list1=[]
transactions=pd.read_excel(path,sheet_name=1)
for i in range(0,len(transactions)):
    list1.append(1)
transactions['n']=list1
# Merge dataframes

#df=pd.concat([offers,transactions],1)
df=pd.merge(offers,transactions, on='Offer #', how='outer')
# Look at the first 5 rows
df.head()


# --------------
# Code starts here

# create pivot table
matrix=df.pivot_table(index='Customer Last Name',columns='Offer #',values='n')

# replace missing values with 0

matrix.fillna(0,inplace=True)
# reindex pivot table
matrix.reset_index(inplace=True)

# display first 5 rows
matrix.head()

# Code ends here


# --------------
# import packages
from sklearn.cluster import KMeans

# Code starts here

# initialize KMeans object

cluster=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
# create 'cluster' column
x=cluster.fit_predict(matrix[matrix.columns[1:]])
matrix['cluster']=x
matrix.head()
# Code ends here



# --------------
# import packages
from sklearn.decomposition import PCA

# Code starts here

# initialize pca object with 2 components
pca=PCA(n_components=2, random_state=0)

# create 'x' and 'y' columns donoting observation locations in decomposed form
x=pca.fit_transform(matrix[matrix.columns[1:]])
y=pca.fit_transform(matrix[matrix.columns[1:]])
matrix['x']=x[:,0]
matrix['y']=y[:,1]
# dataframe to visualize clusters by customer names

clusters=matrix[['Customer Last Name','cluster','x','y']].copy()
# visualize clusters
#matrix.columns
#plt.scatter(x='x', y='y', c='cluster', colormap='viridis')
plt.show()
# Code ends here


# --------------
# Code starts here

# merge 'clusters' and 'transactions'

data=pd.merge(clusters,transactions,on='Customer Last Name')
# merge `data` and `offers`
data=pd.merge(offers,data)
# initialzie empty dictionary

champagne={}
counts=[]
# iterate over every cluster
for i in range(0,5):
    # observation falls in that cluster
    new_df=data[data['cluster']==i]
    # sort cluster according to type of 'Varietal'
    counts=new_df['cluster'].value_counts(ascending=False)
    x={i:counts}
    champagne.update(x)
    # check if 'Champagne' is ordered mostly
    #if counts.index[0]=='Champagne':
        #champagne={i:counts[0]}
cluster_champgane=2

        # add it to 'champagne'

#print(data['cluster'].value_counts())
# get cluster with maximum orders of 'Champagne' 

print(champagne)
# print out cluster number




# --------------
# Code starts here

# empty dictionary
discount={}

# iterate over cluster numbers
for i in range(0,5):

    # dataframe for every cluster
    new_df=data[data['cluster']==i]
    # average discount for cluster
    sum1=new_df['Discount (%)'].sum()
    counts=(sum1/len( new_df))
    # adding cluster number as key and average discount as value 
    x={i:counts}
    discount.update(x)
# cluster with maximum average discount
print(discount)
cluster_discount= max(discount, key=discount.get)
#cluster_discount
# Code ends here


