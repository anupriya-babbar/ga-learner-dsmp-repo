# --------------
# Importing header files
import numpy as np

# Path of the file has been stored in variable called 'path'

#New record
new_record=[[50,  9,  4,  1,  0,  0, 40,  0]]
x=np.asarray(new_record)

#Code starts here
data=np.genfromtxt(path,delimiter=',',skip_header=1)
census=np.concatenate([data,x])
print(census)


# --------------
#Code starts here
#y=np.ndim(census)
#print(y)
age=np.array(census[:,0])
print(age)
max_age=np.max(age)
min_age=np.min(age)
age_mean=np.mean(age)
age_std=np.std(age)
print(max_age,min_age,age_mean,age_std)


# --------------
#Code starts here



race_0=np.array(census[census[:,2]==0])
print(race_0)
race_1=np.array(census[census[:,2]==1])
race_2=np.array(census[census[:,2]==2])
race_3=np.array(census[census[:,2]==3])
race_4=np.array(census[census[:,2]==4])
print(race_1,race_2,race_3,race_4)
len_0=len(race_0)
len_1=len(race_1)
len_2=len(race_2)
len_3=len(race_3)
len_4=len(race_4)
print(len_0,len_1,len_2,len_3,len_4)

minority_race=3
print(minority_race)
#print(minority_race)



# --------------
#Code starts here
senior_citizens=census[census[:,0]>60]
print(senior_citizens)
working_hours=census[:,6]
working_hours_sum=senior_citizens.sum(axis=0)[6]
senior_citizens_len=len(senior_citizens)
avg_working_hours=working_hours_sum/senior_citizens_len
print(avg_working_hours)




# --------------
#Code starts here
high=census[census[:,1]>10]
low=census[census[:,1]<=10]
avg_pay_high=high.mean(axis=0)[7]
avg_pay_low=low.mean(axis=0)[7]
print(avg_pay_high,avg_pay_low)
if(avg_pay_high>avg_pay_low):
    print("more study-more pay")
else:
    print("no better education leads to better pay")



