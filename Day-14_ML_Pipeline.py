# Importing the dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle

# ========================================================================================================================== #
def main():
    # Aplying the styles
    plt.style.use("dark_background")

    # Importing the dataset
    df = pd.read_csv("housing.csv")

    # Extracting info from the data
    print("--- Data Info ---")
    df.info()
    print("\n--- Missing Values Before Cleaning ---")
    print(df.isnull().sum())

    # Filling missing values with median
    df["total_bedrooms"].fillna(
        df["total_bedrooms"].median(),
        inplace=True
)

    # Plotting scatter plot
    plt.scatter(
        df["longitude"], df["latitude"], c=df["median_house_value"], cmap="jet", s=5
    )

    # Labelling of the scatter plot
    plt.title("House Prices based on Location")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Scatter plot show
    plt.grid(alpha=0.5)
    plt.show()

    # Plotting a histogram
    plt.hist(df["median_house_value"], bins=50, color="maroon")

    # Title labelling
    plt.title("Distribution of Median House Values")

    # Histogram show
    plt.grid(alpha=0.5)
    plt.show()

    # Plotting the heatmap
    pivot = pd.pivot_table(
        df, values="median_house_value", index="ocean_proximity", aggfunc="mean"
    )
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="magma")

    # Labelling and plotting the heatmap
    plt.title("Average House Value by Ocean Proximity")
    plt.show()

# ========================================================================================================================== #

# --- Data Preprocessing --- #

    # Defining X (Features) and Y (Target)
    x = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    # ColumnTransformer configuration (Scaler for numbers, Encoder for text)
    ct = ColumnTransformer(
        transformers=[
        (
            "num_scaler",
            StandardScaler(),
            [
                "longitude",
                "latitude",
                "housing_median_age",
                "total_rooms",
                "total_bedrooms",
                "population",
                "households",
                "median_income",
            ],
            ),
            ("cat_encoder", OneHotEncoder(sparse_output=False), ["ocean_proximity"]),
        ],
        remainder="passthrough",
    )

    # Train-Test Split (20% test size is standard)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=3,
    )

    # Applying transformation to training and testing data
    x_train_transformed = ct.fit_transform(x_train)
    x_test_transformed = ct.transform(x_test)

# ========================================================================================================================== #

# --- Model Training --- #
    model = LinearRegression()
    model.fit(x_train_transformed, y_train)

# ========================================================================================================================== #

# --- Model Prediction & Evaluation --- #
    y_pred = model.predict(x_test_transformed)

    print("\n--- Model Evaluation ---")
    print(f"R2 Score : {r2_score(y_test, y_pred):.4f}")
    print(f"MAE      : {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"MSE      : {mean_squared_error(y_test, y_pred):.2f}")
    print(f"RMSE     : {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# ========================================================================================================================== #

    # Actual vs Predicted Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color="cyan")
    plt.xlabel("Actual House Value")
    plt.ylabel("Predicted House Value")
    plt.title("Actual vs Predicted House Prices")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")  # Reference line
    plt.grid(True, alpha=0.3)
    plt.show()

# ========================================================================================================================== #

# --- Saving the Model ---#
    with open("house_price_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("preprocessor.pkl", "wb") as f:
        pickle.dump(ct, f)

    print("\nModel and Preprocessor saved successfully.")

# --- Calling the function ---#
if __name__ == "__main__":
    main()
# ========================================================================================================================== #
#                                                 0 END OF THE CODE 1                                                        #
# ========================================================================================================================== #
