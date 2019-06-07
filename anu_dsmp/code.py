# --------------
import pandas as pd
import scipy.stats as stats
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')
#Sample_Size
sample_size=2000

#Z_Critical Score
z_critical = stats.norm.ppf(q = 0.95)  
confidence_interval=[1,2]

# path        [File location variable]
data = pd.read_csv(path)
#Code starts here
data_sample=data.sample(n=sample_size,random_state=0)
sample_mean=data_sample['installment'].mean()
sample_std=data_sample['installment'].std()
pop_std=data['installment'].std()
margin_of_error=z_critical*(sample_std/math.sqrt(sample_size))
confidence_interval[0]=sample_mean-margin_of_error
confidence_interval[1]=sample_mean+margin_of_error
true_mean=data['installment'].mean()
if true_mean>confidence_interval[0]and true_mean<confidence_interval[1]:
    print('Falls')
else:
    print('Rejected')


# --------------
import matplotlib.pyplot as plt
import numpy as np

#Different sample sizes to take
sample_size=np.array([20,50,100])
fig, axes = plt.subplots(nrows=3, ncols=1)
#Code starts here
for i in range (len(sample_size)):
    m=[]
    for j in range (1,1000):
        sample_data=data.sample(n=sample_size[i],random_state=0)
        sample_mean=sample_data['installment'].mean()
        m.append(sample_mean)
    
    mean_series=pd.Series(m)
    axes[i]=plt.hist(mean_series)


# --------------
#Importing header files

from statsmodels.stats.weightstats import ztest

#Code starts here
data['int.rate']=data['int.rate'].str.replace('%','')
data['int.rate']=data['int.rate'].astype(float)
data['int.rate']=data['int.rate']/100
x1=data[data['purpose']=='small_business']['int.rate']
#value=data['int.rate'].mean()
z_statistic,p_value=ztest(x1,value=data['int.rate'].mean(),alternative='larger')
print(z_statistic,p_value)
if p_value< .05:
    print("reject null hypothesis")
else:
    print("Cant reject null hypothesis")


# --------------
#Importing header files
from statsmodels.stats.weightstats import ztest


#Code starts here
x1=data[data['paid.back.loan']=='No']['installment']
x2=data[data['paid.back.loan']=='Yes']['installment']
z_statistic,p_value = ztest(x1,x2)
if p_value< 0.05:
    print("reject null hypothesis")
else:
    print("cant reject null hypothesis")



# --------------
#Importing header files
from scipy.stats import chi2_contingency

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1

#Code starts here
yes=data[data['paid.back.loan']=='Yes']['purpose'].value_counts()
no=data[data['paid.back.loan']=='No']['purpose'].value_counts()
#print(no.notnull().count())
#yes=pd.Series(yes['purpose'])
#no=pd.Series(no['purpose'])
#observed=pd.crosstab(yes,no)
observed=pd.concat([yes.transpose(),no.transpose()],1,keys=['Yes','No'])
#type(observed)
chi2,p,dof,ex=chi2_contingency(observed)

#print(observed)


