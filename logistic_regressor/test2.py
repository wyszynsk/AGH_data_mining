from regresion import LogisticRegressor
from sklearn.linear_model import LogisticRegression
from utils import acctual_vs_pred, plot
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

X_2, y_2 = make_classification(n_samples=1000,   
                           n_features=2,    
                           n_informative=2, 
                           n_redundant=0,   
                           n_clusters_per_class=1,
                           random_state=3)

plot(X_2, y_2)

X_train, X_test, y_train, y_test = train_test_split(X_2, y_2, test_size=0.25, random_state=42)

model = LogisticRegressor(learning_rate = 0.5, iterations = 1000) #learning rate 0.5 daje najlepsze accuracy
model.fit(X_train, y_train)  

y_pred = model.predict(X_test)  

model_sk = LogisticRegression()
model_sk.fit(X_train, y_train)

y_pred_sk = model_sk.predict(X_test)

accuracy_mine = accuracy_score(y_test, y_pred)
accuracy_sk = accuracy_score(y_test, y_pred_sk)
print(f'My model accuracy: {accuracy_mine:.2f}, Sklearn model accuracy: {accuracy_sk:.2f}')

acctual_vs_pred(X_test, y_test, y_pred)