from regresion import LogisticRegressor
from sklearn.linear_model import LogisticRegression
from utils import acctual_vs_pred, plot
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=100,   
                           n_features=2,    
                           n_informative=2, 
                           n_redundant=0,   
                           n_clusters_per_class=1,
                           random_state=42)

plot(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# training my model
model = LogisticRegressor(learning_rate = 0.05, iterations = 1000) #learning rate 0.05 daje 100%acc
model.fit(X_train, y_train)  

y_pred = model.predict(X_test)  

model_sk = LogisticRegression()
model_sk.fit(X_train, y_train)

y_pred_sk = model_sk.predict(X_test)

accuracy_mine = accuracy_score(y_test, y_pred)
accuracy_sk = accuracy_score(y_test, y_pred_sk)
print(f'My model accuracy: {accuracy_mine:.2f}, Sklearn model accuracy: {accuracy_sk:.2f}')

acctual_vs_pred(X_test, y_test, y_pred)