import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset=pd.read_csv('games.csv')

y=dataset['average_rating']


dataset.drop(['name','type','id','average_rating','yearpublished','maxplaytime','users_rated','bayes_average_rating','total_comments',],axis=1,inplace=True)
dataset.fillna(dataset.mean(),inplace=True)

x_train,x_test,y_train,y_test=train_test_split(dataset,y,test_size=0.2)

"""

for i in range(len(dataset.columns[:5])): 
    plt.figure()
    plt.bar(dataset.iloc[:,i],y)
    plt.title(dataset.columns[i])
    
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


model_randomforest=RandomForestRegressor(n_estimators=100)


model_randomforest.fit(x_train,y_train)    
y_pred=model_randomforest.predict(x_test)
print(mean_squared_error(y_test,y_pred))
print(r2_score(y_test,y_pred))




from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


model_linearreg=LinearRegression()


model_linearreg.fit(x_train,y_train)    
y_pred=model_linearreg.predict(x_test)
print(mean_squared_error(y_test,y_pred))
print(r2_score(y_test,y_pred))