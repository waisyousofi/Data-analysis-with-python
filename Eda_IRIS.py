"""
Created on Fri Dec 18 16:08:40 2020

@author: Waisullah yousofi
"""
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ttest_1samp

df = pd.read_excel("./Data/iris.xlsx")
df.head()


### ***********************************#
#1. What is structure of the dataset.
#**********************************#
print(df.shape)

#as we can see it has 150 rows and 6 column which are : Id, SepalLength, SeplalWdith, PetalLength, PetalWdith and species. 
print(df.columns)


#*****************************************#
#2. What are the data type of each columns?
#*****************************************#

#after running below command we can see that; Id is integer datatype,
#SepalLength, SeplalWdith, PetalLength, PetalWdith are float datatype and species is object datatype 
print(df.info())


#**********************************************#
#3. What is the length of alpha numeric columns?
#**********************************************#

#the length of alpha numeric column is the length of the largest alphanumeric value in that column.



maxvalue=df['species'].max()
print("Name Of ALpha-Numeric Columns is:",maxvalue)
print("")
print("Leghth of columns is: ",df.species.str.isalnum().sum())


#************************************************#
#4. What are precision & scale of numeric columns?
#************************************************#
#The precision of a numeric is the total count of significant digits in the whole number, that is, the number of 
#digits to both sides of the decimal point. The scale of a numeric is the count of decimal digits in the fractional part,
# to the right of the decimal point. So the number 23.5141 has a precision of 6 and a scale of 4. Integers can be 
#considered to have a scale of zero.

#Since Column 'Id' is of type Integer that is why its considered to have a scale of zero.
#and the Precision is:
maxid=df['Id'].max()
strid=len(str(maxid))
print("Precision of column Id is:",strid,"\tvalue= ",maxid)

#list of other numeric columns:
    
numcol_list=[]    
for numcols in df.columns:
    if(numcols=='species' or numcols=='Id'):
        continue
    else:
        numcol_list.append(numcols)
print("\n",numcol_list)        

def percision_Scale(col):
    prevs=0
    currt=0
    for b in df[col]:
        prevs=currt
        currt=len(str(b))
        if(currt>prevs):
            maxlength=currt
            element=str(b) 
    frac_digits=element.split('.')[1]
    print(col, ": value= ",element,"\nPrecision=",maxlength-1," Precision value=",element,"\tScale: ",len(frac_digits),"\tScale value: ",frac_digits,"\n")
            

for numcolitem in numcol_list:
    percision_Scale(numcolitem)
    


#***********************************#
#5. What are the significant columns?
#***********************************#
"""

In a dataset significant columns are those which provide important informatiosn on
which we can perform different operations.
In this dataset Id is not a significant column because it doesn't provide relative 
informations regargingflowers where as columns: SepalLength, SepalWidth, PetalLength ,
species and PetalWidth are significant columns because they give relative and usefull 
information about flowers. for example based on species wecan classify those flowers. 

"""


#**********************************************#
# 6. Identify significant columns of the dataset.
#**********************************************#
sigcol_list=[]    
for scols in df.columns:
    if(scols!='Id'):
        sigcol_list.append(scols)        
print("\n",sigcol_list)        



#****************************************#
# 7. Find out for each significant column
# ▪ Number of Null values
# ▪ Number of zeros
#****************************************#

#***** Number of Null value *****#
print("#*****Number of Zeros*****#")
print(df.isnull().sum())
# print(df.any())
print("")
print("*****Number of Zeros*****")
print((df==0).sum())


#******************************#
# 8. For each significant column
# ▪ Provide the obvious errors
#******************************#

print("*** obvious errors ***")
print(df.groupby(['species'])[['SepalLength']].count())
print("")


#***********************************************************#
# 9. For each numeric column
# ▪ Replace null values & zeros with mean value of the column.
#***********************************************************#
print("**Before replacing null values and zeros with the mean value of respective columns**")
print((df.isnull()).sum())
print((df==0).sum())

a = df['SepalLength'].mean()    
a=round(a,1)
print("Sepal Legth Mean",a)

b = df['SepalWidth'].mean()
b=round(b,1)
print("Sepal Width Mean =",b)

c = df['PetalLength'].mean()
c=round(c,1)
print("Petal Length Mean",c)

d = df['PetalWidth'].mean()    
d=round(d,1)     
print("Petal Width Mean",d)


df['SepalLength'] = np.where( (df['SepalLength'].isnull()) | (df['SepalLength']==0),a,df['SepalLength']) 
df['SepalWidth'] = np.where( (df['SepalWidth'].isnull()) | (df['SepalWidth']==0),b,df['SepalWidth']) 
df['PetalLength'] = np.where( (df['PetalLength'].isnull()) | (df['PetalLength']==0),c,df['PetalLength']) 
df['PetalWidth'] = np.where( (df['PetalWidth'].isnull()) | (df['PetalWidth']==0),d,df['PetalWidth']) 

print("**Checking the changes after replacing null values and zeros with the mean value of respective column**")
print((df.isnull()).sum())
print((df==0).sum())



#***************************************************************#
# 10. For each significant column
# ▪ Provide the quartile summary along with the count, mean & sum
#***************************************************************#
df.describe()



#**************************************************#
#11. For each significant column
#▪ Provide the range, variance and standard deviation
#**************************************************#

print("*****range ******")
print(df.iloc[:,1:5].max()-df.iloc[:,1:5].min())

#*****************************VARIANCE********************************#
print("*****VARIANCE******")
print(df[['SepalLength','SepalWidth','PetalLength','PetalWidth']].var())

#*************************STANDARD DEVIATION****************************#
print("*****STANDARD DEVIATION******")
print(df[['SepalLength','SepalWidth','PetalLength','PetalWidth']].std())


###############################################
#12. For each significant column
#▪ Provide the count of outliers and their value
###############################################

import utils
# check outlier index
print('\n*** Outlier Index ***')
print(utils.OutlierIndex(df))

# check outlier values
print('\n*** Outlier Values ***')
print(utils.OutlierValues(df))



#**********************************************************#
#13. Are there any class variables? If yes,
#▪ provide frequency distribution table & chart for the same
#**********************************************************#
import warnings
warnings.filterwarnings('ignore')

# create list of class variable manually
colNames = df[['species']]

for colName in colNames:
    print("\n*"+colName+"*")
    print(df.groupby(colName).size())
    print("")
    
    
#Chart of Distribution table
plt.figure()
plt.suptitle("Chart of distribution table")
ax = sns.countplot(df[colName],label="Count")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.grid()
plt.show()




#****************************#
#14. For all numeric columns
#▪ Provide histogram
#****************************#
columns = df[['SepalLength','SepalWidth','PetalLength','PetalWidth']]
plt.figure()
plt.suptitle('Histogram of numberic columns')
sns.histplot(data=columns,kde=False)
plt.grid()
plt.show()


##############################
#15. For all numeric variables
#▪ Provide box & whisker plot
##############################

columns = df[['SepalLength','SepalWidth','PetalLength','PetalWidth']]

# quan = columns.quantile(0.95)
# maxim = df[columns >quan]
# maxim.sum()
plt.figure(figsize=(10,5))
plt.suptitle('Box & Whisker Plot')
sns.boxplot(data=columns)
plt.grid()
plt.show()


#*******************************#
#16 For all numeric variables
#▪ Provide correlation table & graph
#***********************************#
df.corr()
plt.figure(figsize=(10,5))
plt.title('Heat Map of correlation')
ax = sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")
plt.show()

print("\n*** Correlation Table ***")
print(df.corr())


#***************************************#
# 17. Based on the correlation table
# ▪ Provide scatter plot relevant columns
#***************************************#

#sns.pairplot(data=df,hue='species')
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(16,6))
plt.figure(figsize=(8,5))
plt.suptitle('Scatter plot of relevent columns')

ax1.set_title('Sepal Length vs Width')
sns.scatterplot(x='SepalLength',y='PetalLength',hue='species',data=df,ax=ax1)

ax2.set_title('Petal Length  vs Width')
sns.scatterplot(x='SepalWidth',y='PetalWidth',hue='species',data=df,ax=ax2)

plt.show()



#**********************************************************************#
# 18. Use relevant statistical test to check if population mean value of 
#SepalLength is significantly different from 6.25. Use 0.06 as α(alpha)
#**********************************************************************#

##############################################################
# Hypothesis Test
##############################################################

# Problem: Check if the population mean value of sepallength is significantly differnt from 6.25.

#Ho:	m = 6.25
#Ha:	m != 6.25
#Tail: Two
#a is 0.06

# null hyp
Ho = "mu = 6.25"
# alt hyp
Ha = "mu != 6.25"
# alpha
al = 0.06 #significance  level
# mu - mean
mu = 6.25
# tail type
tt = 2
# data
SepalLength = df['SepalLength'].values
# print
print("Ho:", Ho)
print("Ha:", Ha)
print("al:", al)
print("mu:", mu)

print("")
print("*** Testing Results ***")
print("")

ts, pv = ttest_1samp(SepalLength, mu)
print("t-stat",ts)
print("p-vals",pv)
t2pv = pv
t1pv = pv*2
print("1t pv",t1pv)
print("2t pv",t2pv)

if tt == 1:
    if t1pv < al:
        print("Null Hypothesis: Rejected")
        print("Conclusion:",Ha)
    else:
        print("Null Hypothesis: Failed to Rejected")
        print("Conclusion:",Ho)
else:
    if t2pv < al/2:
        print("Null Hypothesis: Rejected")
        print("Conclusion:",Ha)
    else:
        print("Null Hypothesis: Failed to Rejected")
        print("Conclusion:",Ho)

        
        
#********************************************************************************************************#
#19. Use relevant statistical test to check if the population mean value of PetalWidth is no more than 1.5.
#********************************************************************************************************#


##############################################################
# Hypothesis Test
##############################################################

# Problem: Check if the population mean age is PetalWidth more than 1.5

#Ho:	m <= 1.5
#Ha:	m > 1.5
#Tail: One
#Test: One Sample Mean without std

from scipy.stats import ttest_1samp


# null hyp
Ho = "mu <= 1.5"
# alt hyp
Ha = "mu > 1.5"
# alpha
al = 0.06  #significance  level
# mu - mean
mu = 1.5
# tail type
tt = 1
# data
PetalWidth = df['PetalWidth'].values
# print
print("Ho:", Ho)
print("Ha:", Ha)
print("al:", al)
print("mu:", mu)

print("")
print("*** Testing Results ***")
print("")

ts, pv = ttest_1samp(PetalWidth, mu)
print("t-stat",ts)
print("p-vals",pv)
t2pv = pv
t1pv = pv*2
print("1t pv",t1pv)
print("2t pv",t2pv)

if tt == 1:
    if t1pv < al:
        print("Null Hypothesis: Rejected")
        print("Conclusion:",Ha)
    else:
        print("Null Hypothesis: Failed to Rejected")
        print("Conclusion:",Ho)
else:
    if t2pv < al/2:
        print("Null Hypothesis: Rejected")
        print("Conclusion:",Ha)
    else:
        print("Null Hypothesis: Failed to Rejected")
        print("Conclusion:",Ho)
        
        
#*************** Exploratory Data Analysis *******************#
""" 
                    OVERVIEW
                    
                #Reproduction

Analysis started	2020-12-14-20 16:17:29.249705
Analysis finished	2020-12-20 16:17:36.200341
Duration	        7 Days
python packages 	 Pandas,Numpy,Matplotlib,Seaborn,Scipy
IDE 	             Spyder , Jupyter Notebook


               #Dataset statistics

Number of variables	                 6
Number of observations	             150
Missing cells	                     4
Missing cells (%)	                 0.4%
Duplicate rows	                     0
Duplicate rows (%)	                 0.0%
Total size in memory	             7.2 KiB
Average record size in memory	     48.9 B

               #VARIABLE TYPE
                    
Numerical	           0 to 7.9  
Categorical 	       4 (Iris-setosa, setosa ,Iris-versicolor, Iris-virginica)

                            #VARIABLE
                    
                    SepalWidth  PetalLength PetalWidth species
    Distinct	     36             45          23         6
    Missing	1	     1              1           1          0
    Mean	         5.8            3.7         1.1
    Minimum	         0              0           0
    Maximum	         7.9            6.9         2.5
    Zeros	         1              1           1
     
    #Visualization:
      *Distributaion Table :It is indicating that how our data is distributed.
    
      *Box & Whisker Plot: Here we notice that we have 3 out liers in our dataset.
      
    #Co-relation:
        With the help o heat map we observe that the column PetalLength & PetalWidth are Highly Co-related to each other
      
    
    ***Conclusion:***
      Here in EDA we use Hypothesis testing for testing and getting final.
      Where our population mean value of sepallength is not equal to 6.25.
      In other hand we done testing on population mean age is PetalWidth were we find the mean is greater then 1.5
      
      

"""