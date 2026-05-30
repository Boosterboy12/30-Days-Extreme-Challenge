# Importing The Requirements
import pandas as pd
import matplotlib.pyplot as plt

# Making A Hardware Dataset
hardware_data = {
    "Mouse_Model": [
        "Razer DeathAdder",
        "Logitech G502",
        "ASUS Rogue",
        "Razer Viper",
        "Logitech G Pro",
        "Corsair Dark",
        "SteelSeries Prime",
        "ASUS TUF",
    ],
    "Brand": [
        "Razer",
        "Logitech",
        "ASUS",
        "Razer",
        "Logitech",
        "Corsair",
        "SteelSeries",
        "ASUS",
    ],
    "Price_USD": [70, 80, 110, 140, 150, 90, 60, None],  # One missing price
    "Weight_Grams": [88, 121, 110, 58, 63, 142, 69, 85],
    "RGB_Zones": [1, 11, 3, 2, 1, 9, 1, 2],
    "Comfort_Rating": [4.5, 4.8, 4.2, 4.6, 4.7, 4.0, 4.3, 3.9],
}

# Loading the Dataset
df_hardware = pd.DataFrame(hardware_data)

#Filling Null Values
df_hardware["Price_USD"] = df_hardware["Price_USD"].fillna(
    df_hardware["Price_USD"].mean()
)

# Task 1
# Create a normal vertical bar chart showing Mouse_Model on X-axis and Price_USD on Y-axis. Make the bars crimson red.

# Solution 
plt.figure(figsize=(20, 10))
plt.bar(df_hardware["Mouse_Model"], df_hardware["Price_USD"], color="crimson")
plt.title(
    "Famous Mouse Model Prices",
    fontdict={
        "family": "monospace",
        "color": "#08ff31",
        "weight": "bold",
        "size": 18,
    },
)
plt.xlabel(
    "Mouse",
    fontdict={
        "family": "monospace",
        "color": "#ff0000",
        "weight": "bold",
        "size": 12,
    },
)
plt.ylabel(
    "Price",
    fontdict={
        "family": "monospace",
        "color": "#ff0000",
        "weight": "bold",
        "size": 12,
    },
)
plt.grid()
plt.tight_layout()
plt.show()


# Task 2
# Create a horizontal bar chart showing Mouse_Model on Y-axis and Weight_Grams on X-axis to see which mouse is the heaviest.

# Solution 
plt.figure(figsize=(15, 10))
plt.barh(
    hardware_data["Mouse_Model"],
    hardware_data["Weight_Grams"],
    color="#2B2D42",
)
plt.title(
    "Mouse Model Weight Comparison",
    fontdict={
        "color": "green",
        "family": "monospace",
        "size": 20,
        "weight": "bold",
    },
)
plt.ylabel(
    "Mouse Model",
    fontdict={
        "color": "Red",
        "family": "monospace",
        "size": 17,
        "weight": "bold",
    },
)
plt.xlabel(
    "Weight In Grams",
    fontdict={
        "color": "Red",
        "family": "monospace",
        "size": 17,
        "weight": "bold",
    },
)
plt.grid()
plt.tight_layout()
plt.show()


# Task 3
# Create a scatter plot with Weight_Grams on X-axis and Comfort_Rating on Y-axis using large orange dots with grid lines enabled.

# Solution 
plt.figure(figsize=(12,6))
plt.scatter(
    hardware_data["Weight_Grams"], hardware_data["Comfort_Rating"], c="orange", s=100
)
plt.grid()
plt.ylabel(
    "Comfort Rating",
    fontdict={
        "color": "maroon",
        "family": "monospace",
        "weight": "bold",
        "size": 20,
    },
)
plt.xlabel(
    "Weight In Grams",
    fontdict={
        "color": "maroon",
        "family": "monospace",
        "weight": "bold",
        "size": 20,
    },
)
plt.title(
    "Mouse Weight And Comfort Rating Analysis",
    fontdict={
        "color": "black",
        "family": "monospace",
        "weight": "bold",
        "size": 20,
    },
)
plt.tight_layout()
plt.show()
