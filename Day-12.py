# Importing dependencies
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Task-1 : A company has provided employee information. Your goal is to prepare the dataset so that it can be used to train a machine learning model that predicts employee salaries.

# ========================================================================================= #
#                                       Solution                                            #
# ========================================================================================= #

# Creating the dataset
df = pd.DataFrame(
    {
        "Employee_ID": [1, 2, 3, 4, 5],
        "Gender": ["Male", "Female", "Male", "Female", "Male"],
        "Department": ["HR", "IT", "Finance", "IT", "HR"],
        "Age": [25, 30, 35, 28, 40],
        "Experience": [2, 5, 8, 4, 12],
        "Salary": [30000, 50000, 70000, 45000, 90000],
    }
)

# Creating the models & Dataframes 
df.drop("Employee_ID", axis=1, inplace=True)
gender_model = LabelEncoder()
department_model = OneHotEncoder()
age_model = StandardScaler()
experience_model = StandardScaler()

# Transforming the data
gender_df = gender_model.fit_transform(df["Gender"])
department_df = department_model.fit_transform(df[["Department"]])
age_df = age_model.fit_transform(df[["Age"]])
experience_df = experience_model.fit_transform(df[["Experience"]])

# Printing the result 
print(gender_df, department_df, age_df, experience_df)

# Task-2: A school wants to predict whether a student will pass or fail based on the available information.

# ========================================================================================= #
#                                         Solution                                          #
# ========================================================================================= #

# Creating the dataset for the second task
df_challenge_2 = pd.DataFrame({
    "Gender": ["Male", "Female", "Male", "Female", "Male"],
    "Study_Hours": [2, 5, 4, 1, 6],
    "Attendance": [60, 85, 75, 50, 90],
    "City": ["Delhi", "Mumbai", "Chennai", "Delhi", "Mumbai"],
    "Result": ["Fail", "Pass", "Pass", "Fail", "Pass"]
})

# Creating the models 
gender_challenge_2_model = LabelEncoder()
hours_model = StandardScaler()
attendance_model = StandardScaler()
city_model = OneHotEncoder()

# Transforming the data
gender_challenge_2_df = gender_challenge_2_model.fit_transform(df_challenge_2["Gender"])
hours_df = hours_model.fit_transform(df_challenge_2[["Study_Hours"]])
attendance_df = attendance_model.fit_transform(df_challenge_2[["Attendance"]])
city_df = city_model.fit_transform(df_challenge_2[["City"]])

print(gender_challenge_2_df,hours_df,attendance_df,city_df)