# Importing the dependencies
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Task-1: Create a Decision Tree model that predicts whether a student will pass or fail based on the number of hours studied.

# ========================================================================================= #
#                                       Solution                                            #
# ========================================================================================= #
# Making the data
student_data = {"hours_studied": [1, 2, 3, 5, 6, 8], "result": [0, 0, 0, 1, 1, 1]}
x = student_data["hours_studied"]
y = student_data["result"]
x = np.array(student_data["hours_studied"]).reshape(-1, 1)

# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=1,
)

# Training the model
student_model = DecisionTreeClassifier()
student_model.fit(x_train, y_train)

# Predicting
prediction = student_model.predict([[7]])
print(prediction)

# Calculating the accuracy
y_pred = student_model.predict(x_test)
print(accuracy_score(y_test, y_pred))

# Task-2: Build a Decision Tree model to predict whether a player will be selected for a cricket team.

# ========================================================================================= #
#                                        Solution                                           #
# ========================================================================================= #
# Making The Data
x1 = [[13, 1], [14, 1], [15, 0], [13, 0], [14, 1]]
y1 = [1, 1, 0, 0, 1]

# Splitting the data
(
    x1_train,
    x1_test,
    y1_train,
    y1_test,
) = train_test_split(
    x1,
    y1,
    test_size=0.2,
    random_state=1,
)

# Training the model
model_2 = DecisionTreeClassifier()
model_2.fit(x1_train, y1_train)

# Predicting
prediction = model_2.predict([[12, 1]])
print(prediction)

# Accuracy
y1_pred = model_2.predict(x1_test)
print(accuracy_score(y1_test, y1_pred))

# Task-3: Create a Decision Tree model that predicts whether a customer will buy a gaming laptop.

# ========================================================================================= @
#                                       Solution                                            #
# ========================================================================================= #
# Making the data
x2 = [[1, 1], [1, 0], [0, 1], [1, 1], [0, 0]]
y2 = [1, 0, 0, 1, 0]

# Splitting the data
x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2,
    y2,
    test_size=0.2,
    random_state=1,
)

# Training the model
model_3 = DecisionTreeClassifier()
model_3.fit(x2_train, y2_train)

# Predicting
prediction = model_3.predict([[1, 1]])
print(prediction)

# Accuracy
y2_pred = model_3.predict(x2_test)
print(accuracy_score(y2_test, y2_pred))

# ========================================================================================= #
#                                           0 END 1                                         #
# ========================================================================================= #
