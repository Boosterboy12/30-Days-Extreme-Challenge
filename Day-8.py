# Importing Requiremnets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Extracting Data Info
df = pd.read_csv("placement.csv")
print(df.head())
print(df.shape)
df.info()
print(df.describe())
print(df.isnull().sum())

# Extracting The Series
X = df.iloc[:, 0:1]
Y = df.iloc[:, -1]

# Training The Data
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
lr = LinearRegression()
lr.fit(X_train, Y_train)

# Predicting The Values
print(lr.predict(X_test.iloc[[9]]))

# Making The Graph
plt.figure(figsize=(10, 6))
plt.scatter(X_train, Y_train, alpha=0.7)
X_sorted = X_train.sort_values(by="cgpa")
plt.plot(X_sorted, lr.predict(X_sorted), color="red", linewidth=3)

# Labeling
plt.title("CGPA vs Placement Package")
plt.xlabel("CGPA")
plt.ylabel("Package (LPA)")

#  Plotting The Graph
plt.grid(True)
plt.show()

# Evaluating The Model
y_pred = lr.predict(X_test)
print("R2 Score:", r2_score(Y_test, y_pred))
print("MAE:", mean_absolute_error(Y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(Y_test, y_pred)))

# Finale
with open("placement_model.pkl", "wb") as f:
    pickle.dump(lr, f)