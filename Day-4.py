# Importing Requirenents
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Loading The Dataset
data = {
    "Player": [
        "Virat",
        "Rohit",
        "Rahul",
        "Virat",
        "Rohit",
        "Rahul",
        "Hardik",
        "Hardik",
    ],
    "Match_Type": ["T20", "T20", "ODI", "ODI", "T20", "ODI", "T20", "ODI"],
    "Runs": [82, 45, 110, 55, 12, 28, 64, None],
    "Balls": [53, 28, 95, 40, 9, 32, 35, 15],
}
df = pd.DataFrame(data)
# Gathering Information From The Data & Of The Data
print(df.describe())
print(df.info())
# Performing Functions On The Data
# Task 1: Handle the missing value in 'Runs' for Hardik by filling it with 0.
df["Runs"] = df["Runs"].fillna(0)
# Task 2: Filter the data to create a new DataFrame that contains only 'T20' matches.
data = df[df["Match_Type"] == "T20"]
print(data)
# Task 3: find the total 'Runs' scored by each individual 'Player'.
print(df.groupby("Player")["Runs"].sum().sort_values(ascending=False))
# Task 4: Calculate the Strike Rate for every row and save it in a new column named 'Strike_Rate'.
df["Strike_Rate"] = (df["Runs"] / df["Balls"]) * 100
print(df)
# Task 5: Find the missing value (NaN) generated in the 'Strike_Rate' column and fill it
# With the Average (Mean) of that entire column.
df["Strike_Rate"] = df["Strike_Rate"].fillna(df["Strike_Rate"].mean())


# Determining The Batting Style
def batting_style(strike_rate):
    # Condition
    if strike_rate > 130:
        return "Aggressive"
    else:
        return "Anchor"

# Assigning The Function And Craeting Another Column
df["Batter_Style"] = df["Strike_Rate"].apply(batting_style)

# Printing Cross Tabulations Table
print(pd.crosstab(df["Player"], df["Match_Type"]))

# Plotting The Graph
pivot_table = df.pivot_table(
    index="Player", values="Runs", columns="Balls", aggfunc="mean"
)
sns.heatmap(pivot_table,annot=True,fmt=".0f", cmap="rocket")
plt.show()

# Printing The Final Output
print(df)
