# --- Importing the requirements --- #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# ========================================================================================================================================= #
# ========================================================================================================================================= #


def load_and_clean_data():

    # --- PHASE 1: DATA INGESTION & LOADING --- #
    df = pd.read_csv("heart.csv")

    # --- GAINING DATA INFO --- #
    print(df.head())
    df.info()
    print(df.describe())
    print(df.dtypes)

    # --- CHECKING FOR NULL VALUES --- #
    print((df["RestingBP"] == 0).sum())
    print((df["Cholesterol"] == 0).sum())

    # --- CALCULATE MEDIANS FOR COLUMNS WITH REPLACED ZEROS --- #
    restingbp_median = df["RestingBP"].median()
    cholesterol_median = df["Cholesterol"].median()

    # --- FILLING 0 WITH NAN VALUES --- #
    df["RestingBP"] = df["RestingBP"].replace(0, np.nan)
    df["Cholesterol"] = df["Cholesterol"].replace(0, np.nan)

    # --- REPLACING NULL VALUES WITH THE MEDIAN --- #
    df["Cholesterol"].fillna(cholesterol_median, inplace=True)
    df["RestingBP"].fillna(restingbp_median, inplace=True)

    return df

# ======================================================================================================================================= #
# ======================================================================================================================================= #


def graphs(df):

    # --- COUNTPLOT - 1 --- #
    sns.countplot(x="HeartDisease", data=df)
    plt.show()

    # --- HISTOGRAM --- #
    sns.histplot(x="Age", hue="HeartDisease", data=df, multiple="stack")
    plt.show()

    # --- BOXPLOT --- #
    sns.boxplot(data=df, x="HeartDisease", y="Cholesterol")
    plt.show()

    # --- COUNTPLOT - 1 --- #
    sns.countplot(data=df, x="ChestPainType", hue="HeartDisease")
    plt.show()

    # --- HEATMAP --- #
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True)
    plt.show()

# ========================================================================================================================================= #
# ========================================================================================================================================= #


def training_and_predicting(df):

    # --- DEFINING X & Y --- #
    x = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    # --- MAKING THE DATA MACHINE READABLE --- #
    model_df = ColumnTransformer(
        transformers=[
            (
                "encoder",
                OneHotEncoder(),
                [
                    "Sex",
                    "ChestPainType",
                    "RestingECG",
                    "ExerciseAngina",
                    "ST_Slope",
                ],
            ),
        ],
        remainder="passthrough",
    )

    # --- SPLITING THE DATA --- #
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=3,
    )

    # --- TRANSFORMING THE DATA --- #
    x_train_transformed = model_df.fit_transform(x_train)
    x_test_transformed = model_df.transform(x_test)

    # --- STANDARD SCALING --- #
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_transformed)
    x_test_scaled = scaler.transform(x_test_transformed)

    # --- LOGISTIC REGRESSION -- #
    lr_model = LogisticRegression()
    lr_model.fit(x_train_scaled, y_train)

    # --- PREDICTING --- #
    prediction = lr_model.predict(x_test_scaled)
    print(prediction)

    # --- ACCURACY --- #
    print(accuracy_score( y_test , prediction ))
    print(classification_report( y_test , prediction))
    print(confusion_matrix(  y_test , prediction ))

    # --- SAVING THE MODEL --- #
    with open("heart_model.pkl", "wb") as f:
        pickle.dump(lr_model, f)

def main():
    df = load_and_clean_data()
    graphs(df)
    training_and_predicting(df)


# --- CALLING THE FUNCTION -- #
if __name__ == "__main__":
    main()