# Importing dependencies
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Task-1: Build a Logistic Regression model to predict whether a student will pass or fail based on the number of hours studied.

# ========================================================================================= #
#                                        Solution                                           #
# ========================================================================================= #
# Making the data
student_data = {"hours": [2, 3, 4, 5, 6, 7, 8], "result": [0, 0, 0, 0, 1, 1, 1]}
x = student_data["hours"]
y = student_data["result"]
x = np.array(student_data["hours"]).reshape(-1, 1)

# Splitting the data for testing too
(
    x_train,
    x_test,
    y_train,
    y_test,
) = train_test_split(x, y, test_size=0.2, random_state=1)

# Training the model to predict
model = LogisticRegression()
model.fit(x_train, y_train)

# Predicting
prediction = model.predict([[6]])
print(prediction)

# Calculating the accuracy
y_pred = model.predict(x_test)
print(accuracy_score(y_test, y_pred))

# Task-2: Build a Logistic Regression model to predict whether a customer will purchase a product based on their age.

# ========================================================================================= #
#                                        Solution                                           #
# ========================================================================================= #
# Making the data
purchase_data = {
    "Age": [
        7,
        18,
        22,
        25,
        28,
        32,
        35,
    ],
    "Purchase": [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
    ],
}
x1 = purchase_data["Age"]
y1 = purchase_data["Purchase"]
x1 = np.array(purchase_data["Age"]).reshape(-1, 1)

# Splitting the data for testing too
(
    x1_train,
    x1_test,
    y1_train,
    y1_test,
) = train_test_split(x1, y1, test_size=0.2, random_state=1)

# Training the model to predict
model = LogisticRegression()
model.fit(x1_train, y1_train)

# Predicting
prediction = model.predict([[31]])
print(prediction)

# Calculating the accuracy
y1_pred = model.predict(x1_test)
print(accuracy_score(y1_test, y1_pred))

# Task-3: Build a Logistic Regression model to predict whether an employee will be promoted based on years of experience.

# ========================================================================================= #
#                                         Solution                                          #
# ========================================================================================= #
# Making the data
employee_data = {
    "Experience": [1, 2, 3, 5, 7, 8, 10, 12],
    "Promotion": [0, 0, 0, 1, 1, 1, 1, 1],
}
x2 = employee_data["Experience"]
y2 = employee_data["Promotion"]
x2 = np.array(employee_data["Experience"]).reshape(-1, 1)

# Splitting the data for testing too
(
    x2_train,
    x2_test,
    y2_train,
    y2_test,
) = train_test_split(x2, y2, test_size=0.2, random_state=1)

# Training the model to predict
model = LogisticRegression()
model.fit(x2_train, y2_train)

# Predicting
prediction_2 = model.predict([[3]])
print(prediction_2)

# Calculating the accuracy
y2_pred = model.predict(x2_test)
print(accuracy_score(y2_test, y2_pred))
# ========================================================================================= #
#                                         0 END 1                                           #
# ========================================================================================= #
