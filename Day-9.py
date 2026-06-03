# ========================================================================================= #
#                                     By Boosterboy12                                       #
# ========================================================================================= #

# Importing the requirements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Challenge 1: Train a simple regression model on a 75-25 split to predict pizza price using diameter and find the cost for a 15-inch pizza.
# ========================================================================================= #
#                                       Solution                                            #
# ========================================================================================= #


# This includes data making and processing it
def data_ch_1():

    # Making the data
    pizza_data = {
        "Diameter_Inches": [6, 8, 10, 12, 14, 16, 18, 20],
        "Price_Rs": [150, 250, 380, 480, 610, 720, 890, 1020],
    }

    # Extracting and reshaping the data
    X = np.array(pizza_data["Diameter_Inches"]).reshape(-1, 1)
    y = np.array(pizza_data["Price_Rs"])

    # Returning X & y
    return X, y


#  Here is the main training & prediction code
def main_ch_1():
    X, y = data_ch_1()

    # Splitting the data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=2,
    )

    # Training the model
    lr = LinearRegression()
    lr.fit(X_train, Y_train)

    # Predicting the cost for a 15-inch pizza
    pizza_15_inch = np.array([[15]])
    predicted_price = lr.predict(pizza_15_inch)
    print(f"Predicted price for a 15-inch pizza: Rs. {predicted_price[0]:.2f}\n")

    # Returning the values
    return lr, X_train, Y_train


# Graph making and labelling procceses are here
def graph_ch_1(lr, X_train, Y_train):

    # Making the graph
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, Y_train, color="blue", label="Training Data", alpha=0.7)

    # Sorting the graph smoothly
    X_sorted = np.sort(X_train, axis=0)

    # Adding the essential labels to the graph
    plt.plot(
        X_sorted,
        lr.predict(X_sorted),
        color="red",
        linewidth=3,
        label="Regression Line",
    )

    # Labeling
    plt.title("Pizza Price Prediction vs Diameter")
    plt.xlabel("Diameter (Inches)")
    plt.ylabel("Price (Rs)")
    plt.legend()
    plt.grid(True)


# Calling The Function
if __name__ == "__main__":
    model, X_train, Y_train = main_ch_1()
    graph_ch_1(model, X_train, Y_train)

# Challenge 2: Build a multiple regression model using length and CTR to predict views and forecast viewership for a 14-minute video with 8.5% CTR.
# ======================================================================================== # #                                      Solution                                            #
# ======================================================================================== #


# Challenge 2 data and processing it
def data_ch_2():
    # Making the data
    youtube_data = {
        "Video_Length_Min": [5, 12, 8, 22, 15, 6, 18, 10, 25, 4],
        "Thumbnail_CTR_Percent": [4.5, 8.2, 5.0, 9.1, 7.5, 3.8, 6.2, 8.0, 10.5, 5.5],
        "Views": [1200, 5000, 2300, 9500, 6100, 1100, 4800, 4200, 12000, 1500],
    }

    # Converting to dataframe
    df_yt = pd.DataFrame(youtube_data)

    # Extracting and reshaping the data
    X = df_yt[["Video_Length_Min", "Thumbnail_CTR_Percent"]].values
    y = df_yt["Views"].values

    # Returning X & y
    return X, y


# Main code of challene 2 is here
def main_ch_2():
    X, y = data_ch_2()

    # Splitting the data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=2,
    )
    # Training the data
    lr = LinearRegression()
    lr.fit(X_train, Y_train)

    # Predicting for a 14-minute video with 8.5% CTR
    views_14 = np.array([[14, 8.5]])
    predicted_views = lr.predict(views_14)
    print(f"The PREDICTED view is : {predicted_views[0]:.2f}\n")

    # Returing Values
    return lr, X_train, Y_train


# Graph making and labelling it
def graph_ch_2(lr, X_train, Y_train):

    # Making the graph
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train[:, 0], Y_train, color="blue", label="Training Data", alpha=0.7)

    # Sorting
    sort_idx = np.argsort(X_train[:, 0])
    X_sort = X_train[sort_idx]

    # plotting the graph
    plt.plot(
        X_sort[:, 0],
        lr.predict(X_sort),
        color="red",
        linewidth=3,
        label="Regression Line",
    )

    # Labelling the graph
    plt.title("YouTube Views Training Data")
    plt.xlabel("Video Length (Min)")
    plt.ylabel("Views")
    plt.legend()
    plt.grid(True)


# Calling the function
if __name__ == "__main__":  # <--- Yeh change zaroor kar lena
    model, X_train, Y_train = main_ch_2()
    graph_ch_2(model, X_train, Y_train)


# Challenge-3: Clean missing null values from a phone dataset, train a Multiple Linear Regression model using RAM and storage to predict retail prices, and calculate the final RMSE score.
# ========================================================================================= #
#                                        Solution                                           #
# ========================================================================================= #


# Making the data for challenge 3 is done here
def data_ch_3():
    # Making the data for the challenge
    phone_data = {
        "RAM_GB": [
            4,
            6,
            np.nan,
            8,
            4,
            12,
            6,
            np.nan,
            8,
            12,
        ],
        "Storage_GB": [64, 128, 128, 128, 64, 256, 64, 256, 256, 512],
        "Price_Rs": [
            12000,
            16500,
            15000,
            21000,
            11500,
            35000,
            14000,
            24000,
            26000,
            45000,
        ],
    }

    # Converting to dataframe
    df_ph = pd.DataFrame(phone_data)

    # Filling the null values
    median_ram = df_ph["RAM_GB"].median()
    df_ph["RAM_GB"] = df_ph["RAM_GB"].fillna(median_ram)

    # Extracting and reshaping the data
    X = df_ph[["RAM_GB", "Storage_GB"]].values
    y = df_ph["Price_Rs"].values

    # Returning X & y
    return X, y


# Final main code of challenge 3
def main_ch_3():
    X, y = data_ch_3()

    # Splitting the data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=2,
    )

    # Training the model
    lr = LinearRegression()
    lr.fit(X_train, Y_train)

    # Predicting the price
    price = np.array([[8, 256]])
    predicted_price = lr.predict(price)
    print(f"The PREDICTED price is : {predicted_price[0]:.2f}\n")

    # Returning the values
    return lr, X_train, Y_train


# Making The Graph of challenge 3
def graph_ch_3(lr, X_train, Y_train):

    # Making the graph
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train[:, 0], Y_train, color="blue", label="Training Data", alpha=0.7)

    # Sorting
    sort_idx = np.argsort(X_train[:, 0])
    X_sort = X_train[sort_idx]

    # Plotting the graph
    plt.plot(
        X_sort[:, 0],
        lr.predict(X_sort),
        color="red",
        linewidth=3,
        label="Regression Line",
    )
    # Labelling
    plt.title("Phone Price Prediction vs RAM")
    plt.xlabel("RAM (GB)")
    plt.ylabel("Price (Rs)")
    plt.legend()
    plt.grid(True)
    plt.show()


# Calling out the function
if __name__ == "__main__":  # <--- Yeh change zaroor kar lena
    model, X_train, Y_train = main_ch_3()
    graph_ch_3(model, X_train, Y_train)

# ========================================================================================= #
#                                         0 END 1                                           #
# ========================================================================================= #
