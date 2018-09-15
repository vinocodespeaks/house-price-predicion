
#import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from sklearn.cross_validation import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as stat
from sklearn.metrics import explained_variance_score



#import the dataset
dataset = pd.read_csv("kc_house_data.csv")
#split dependent independent variable  as  x,y
X = dataset.iloc[:,3:]
Y=  dataset.iloc[:,2]

# backward elimination of features using student -T testing
X_ols = np.append(arr=np.ones((21613,1)).astype(int), values=X,axis=1)
backward_X=X_ols[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
regressor_OLS=stat.OLS(endog=Y,exog=X_ols).fit()
regressor_OLS.summary()
#x5>0.05 so we have to remove and fit least square again
#backward_X=np.delete(X_ols,5,axis=1)
backward_X=X_ols[:,[0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18]]
regressor_OLS=stat.OLS(endog=Y,exog=backward_X).fit()
regressor_OLS.summary()
#x4>0.05 so we have to remove and fit least square again
backward_X=X_ols[:,[0,1,2,3,6,7,8,9,10,11,12,13,14,15,16,17,18]]
regressor_OLS=stat.OLS(endog=Y,exog=backward_X).fit()
regressor_OLS.summary()
#set the cleaned X 
X=backward_X[:,1:]
#fit the linear regression model

#splitting the dataset into traning and test set
train_X,test_X,train_Y,test_Y =train_test_split(X,Y,test_size=0.3,random_state=0)

#feature scaling 

sc_X = StandardScaler()
train_X = sc_X.fit_transform(train_X)
test_X = sc_X.transform(test_X)


#fitting the regeression of multiple indipendent variables 
linear_regression = LinearRegression()
linear_regression.fit(train_X,train_Y)
X_hat= linear_regression.predict(test_X)

#accuracy score
accuracy=linear_regression.score(train_X,train_Y)
print(accuracy)




 


  

