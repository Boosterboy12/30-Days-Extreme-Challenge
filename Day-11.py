# Importing the dependencies
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Task-1: Create a Decision Tree model that predicts whether a student will pass or fail based on the number of hours studied.

# ========================================================================================= #
#                                       Solution                                            #
# ========================================================================================= #
# Making the data 
student_data = {
"hours_studied": [1, 2, 3, 5, 6, 8],
"result": [0, 0, 0, 1, 1, 1]
}
x = student_data["hours_studied"]
y = student_data["result"]
x = np.array(student_data["hours_studied"]).reshape(-1, 1)

# Splitting the data
(x_train,x_test,y_train,y_test)=  train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=1,
)

# Training the model
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
