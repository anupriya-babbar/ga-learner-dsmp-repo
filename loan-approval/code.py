# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 



# code starts here
bank=pd.read_csv(path)
categorical_var=bank.select_dtypes(include='object')
print(categorical_var)
numerical_var=bank.select_dtypes(include='number')
print(numerical_var)



# code ends here


# --------------
# code starts here


#code ends here
banks=bank.drop('Loan_ID',axis=1)
print(banks.isnull().sum())
bank_mode=banks.mode()
print(bank_mode)
for x in banks.columns.values:
    banks[x]=banks[x].fillna(value=bank_mode[x].iloc[0])

print(banks)




# --------------
# Code starts here

avg_loan_amount=pd.pivot_table(banks,index=['Gender','Married','Self_Employed'],values='LoanAmount',
aggfunc='mean')
print(avg_loan_amount)

# code ends here



# --------------
# code starts here
#single_type_legendary=len(df[df['Type 2'].isnull() & df['Legendary']==True])
loan_approved=banks[banks['Self_Employed']=='Yes']
loan_approved_s=loan_approved[loan_approved["Loan_Status"]=='Y']
loan_approved_se= len(loan_approved_s)
loan_dis=banks[banks['Self_Employed']=='No']
loan_approved_ns=loan_dis[loan_dis['Loan_Status']=="Y"]
loan_approved_nse= len(loan_approved_ns)
Loan_Status=614
percentage_se=((loan_approved_se/Loan_Status)*100)
print(percentage_se)
percentage_nse=((loan_approved_nse/Loan_Status)*100)
print(percentage_nse)
# code ends here


# --------------
# code starts here


loan_term=banks['Loan_Amount_Term'].apply(lambda x: (int(x) /12))
#print(loan_term)
#y=[]
#loan_term=loan_term.apply(lambda x: y.append(x)  if x>=25 else None)
#print(loan_term)

big_loan_term=len(loan_term[loan_term>=25])

#banks['z']=banks[banks['z']=='True']
#big_loan_term=len(banks['z'])
print(big_loan_term)


# code ends here


# --------------
# code starts here
loan_groupby=banks.groupby(['Loan_Status'])
loan_groupby=banks.groupby('Loan_Status')[['ApplicantIncome','Credit_History']]
mean_values=loan_groupby.mean()
print(mean_values)


# code ends here


