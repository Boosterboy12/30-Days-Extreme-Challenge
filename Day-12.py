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
df.drop("Employee_ID", axis=1, inplace=True)
gender_model = LabelEncoder()
department_model = OneHotEncoder()
age_model = StandardScaler()
experience_model = StandardScaler()

gender_df = gender_model.fit_transform(df["Gender"])
department_df = department_model.fit_transform(df[["Department"]])
age_df = age_model.fit_transform(df[["Age"]])
experience_df = experience_model.fit_transform(df[["Experience"]])

print(gender_df, department_df, age_df, experience_df)
