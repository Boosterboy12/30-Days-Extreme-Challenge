# Importing Requirements
import pandas as pd

# Loading The Dataset
df = pd.read_csv("matches.csv")

# Performing Tasks On The Data
def filtering_data_ipl():
# Task-1: Toss Decision Impact
# Question: Does choosing to field or bat after winning the toss increase win chances?
    print(df[["winner", "toss_winner"]].value_counts().sort_values())
    print("teams winning the toss and choosing field win the most")

# Task-2: MI Performance
# Question: How many matches has 'Mumbai Indians' won in total?
    filtered_df = df[(df["winner"] == "Mumbai Indians")]
    print(len(filtered_df))

# Task-3: Venue Popularity
# Question: Which stadium has hosted the highest number of matches?
    filtered_df1 = df["venue"].value_counts().idxmax()
    print(filtered_df1)

# Task-4: Winning Margin
# Question: Which match witnessed the largest winning margin in terms of runs?
    max_margin = df["result_margin"].max()
    print("Biggest winning margin in runs:", max_margin)

# Task-5: Unique Participation
# Question:Which teams have participated in the tournament so far?
    print(df.drop_duplicates(subset=["team1", "team2"]).shape)

# Executing The Code:
filtering_data_ipl()