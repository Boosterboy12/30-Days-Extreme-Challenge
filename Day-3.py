# Importing Pandas
import pandas as pd

#Loading The Dataframe
df = pd.read_csv("tested.csv")

#Performing Functions on The Dataframe
df.isnull().sum().sum() == 0
df.fillna(0, inplace=True)
print(df["Survived"])
print(df.head())
print(df.tail())

#Print The Final Dataframe
print(df)