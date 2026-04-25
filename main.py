from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

X=df.copy()
X=X.drop("Profit",axis=1)
y=df["Profit"]


X=pd.get_dummies(X,drop_first=True)


(X_train,X_test,y_train,y_test)=train_test_split(X,y,test_size=0.2)

model=Ridge()

model.fit(X_train,y_train)

y_pred=model.predict(X_test)



print(y_pred)


print(r2_score(y_test,y_pred))

print(mean_squared_error(y_test,y_pred))

results=X_test.copy()
results["y_test_actual"]=y_test.values
results["y_yest_predicted"]=y_pred

print(results)

X_test2={"R&D Spend":165349.20,"Administration":136897.80,"Marketing Spend":471784.10,"State_Florida":False,"State_New York":True}

X_test2=pd.DataFrame([X_test2])

y_pred2=model.predict(X_test2)

print(y_pred2)


