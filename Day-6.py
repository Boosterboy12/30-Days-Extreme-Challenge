# ================================================================================= #
#                          File Start This Is By BoosterBoy12                       #
# ================================================================================= #

# Importing The Requiremnets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Using Style
plt.style.use("dark_background")

# LEVEL 1 — FOUNDATION TASKS
# Task 1: Create a line chart showing the daily temperature of a city for 30 consecutive days.

# Solution

# Making The Dataset
days = list(range(1, 31))

temperature = [
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    34,
    33,
    32,
    31,
    30,
    29,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    35,
    34,
    33,
    32,
    31,
    30,
    29,
]

# Plottting The Graph
fig, ax = plt.subplots(figsize=(10, 5))

# Styling The Line
ax.plot(days, temperature, color="#E31A1C", linewidth=3)

# Styling The Title
ax.set_title(
    "Delhi Temperature On 30 consecutive Days", color="#FFFFFF", fontsize=15, pad=10
)

# Styling The Axises
ax.set_xlabel("Days", color="#FFFFFF", fontsize=10, labelpad=5)
ax.set_ylabel("Temperature", color="#FFFFFF", fontsize=10, labelpad=5)

# Styling The Params
ax.tick_params(color="#FFFFFF", labelsize=10)

# Removing Borders
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)  # Remove unnecessary borders
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("#DBD9D9")

# Making It More Systamatic
ax.grid(True, linestyle="--", alpha=0.6, color="#E5E5E5")
plt.tight_layout()
plt.show()

# Task 2: Plot two stock prices on the same graph with proper legends, labels, and gridlines.
# Solution

# Making The Dataset
days = list(range(1, 16))
stock_a = [100, 102, 105, 104, 108, 110, 115, 117, 120, 118, 122, 125, 128, 130, 135]
stock_b = [90, 91, 93, 95, 94, 97, 99, 101, 103, 105, 107, 110, 112, 115, 118]

# Styling the Lines
plt.plot(
    days,
    stock_a,
    color="yellow",
    label="Stock-A",
    linewidth=3,
    marker="o",
    markersize=10,
)
plt.plot(
    days,
    stock_b,
    color="green",
    label="Stock-B",
    linewidth=3,
    marker="o",
    markersize=10,
)

# Styling The Title
plt.title(
    "Stock Prices On Different Days",
    color="Red",
    weight="bold",
    family="monospace",
    size=15,
)

# Styling The Lables
plt.xlabel("Prices", color="White", size=10, weight="bold")
plt.ylabel("Days", color="White", size=10, weight="bold")

# Plotting The Graph
plt.legend(fontsize=15)
plt.grid(color="grey", linestyle="dotted")
plt.tight_layout()
plt.show()

# Task 3: Create a bar chart comparing the sales performance of 10 different products.
# Solution

# Making The Data
products = [
    "Mouse",
    "Keyboard",
    "Monitor",
    "Laptop",
    "Headset",
    "Webcam",
    "Speaker",
    "SSD",
    "RAM",
    "GPU",
]
sales = [120, 95, 60, 40, 80, 50, 70, 90, 110, 30]

# Styling The Bars
plt.bar(products, sales, color="maroon")

# Styling The Title
plt.title("Prices Of Different Products", color="#FFFFFF", size=15)

# Styling The Lables
plt.xlabel("Products", color="pink", weight="bold", size=10)
plt.ylabel("Prices", color="pink", weight="bold", size=10)

# Plotting The Graph
plt.grid(linestyle="dotted", color="grey")
plt.tight_layout()
plt.show()

# Task 4: Visualize the percentage distribution of laptop components using a pie chart.
# Solution

# Making The Data
components = ["CPU", "GPU", "RAM", "Storage", "Battery"]
percentage = [25, 30, 15, 20, 10]

# Styling The Pie Chart
plt.pie(percentage, labels=components, autopct="%0.1f%%")
plt.show()

# LEVEL 2 — ADVANCED VISUALIZATION TASKS
# Task 6: Build a dashboard containing a line chart, bar chart, pie chart, and histogram in a single figure.
# Solution

# Making The Data
months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
sales = [1200, 1400, 1350, 1600, 1800, 2100, 2400, 2200, 2500, 2700, 2900, 3200]
expenses = [800, 900, 850, 1000, 1100, 1300, 1400, 1350, 1500, 1600, 1700, 1800]

# Plotting The Graphs
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))

# Plotting And Styling The 1st Graph
ax[0, 0].plot(sales, expenses, color="red", linewidth=3, marker="o", markersize=10)
ax[0, 0].set_title("Sales VS. Expenses Line Plot")


# Plotting And Styling The 2nd Graph
ax[0, 1].bar(
    months,
    sales,
    color="white",
)
ax[0, 1].bar(
    months,
    expenses,
    color="white",
)

# Plotting The 3rd Graph
ax[1, 0].hist(expenses, bins=5)
ax[1, 0].set_title("Expenses Distribution")

# Plotting And Styling The 4th Graph
ax[1, 1].pie(sales, labels=months, autopct="%0.1f%%")
plt.show()

# Task 7: Generate 5000 random numbers and compare their distributions using multiple histograms.
# Solution

# Making The Dataset
dataset1 = np.random.normal(50, 10, 5000)
dataset2 = np.random.normal(60, 15, 5000)

# Plotting The Graph
plt.hist(dataset1, bins=5)
plt.hist(dataset2, bins=20, alpha=0.5, label="Dataset 2")
plt.title("A Simple Histogram")
plt.show()

# Task 8: Create a scatter plot showing study hours versus exam scores with annotations for top students.
# Solution

# Creating The Dataset
students = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
study_hours = [2, 3, 4, 5, 6, 7, 8, 4, 6, 9]
marks = [45, 52, 58, 65, 72, 80, 85, 60, 75, 95]

# Plotting The Graph
plt.scatter(study_hours, marks, color="red")
plt.axhline(70)
plt.axvline(5.5)
plt.text(9, 95, "Topper1")
plt.text(7, 80, "Topper2")
plt.text(8, 85, "Topper3")
plt.text(6, 72, "Topper4")
plt.text(6, 75, "Topper5")

# Plotting The Graph
plt.show()

# Task 9: Visualize monthly rainfall data using both line and bar charts in the same figure.
# Solution

# Making The Data
months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
rainfall = [10, 15, 20, 35, 70, 120, 180, 170, 140, 90, 40, 20]

# Making The Sublplot
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

# Giving Name To The Title
ax[0].plot(months, rainfall)
ax[1].bar(months, rainfall)
ax[0].set_title("Rainfall In Months Comparison")
ax[1].set_title("Rainfall In Months Comparison")

# Giving The Name To Labels
ax[0].set_xlabel("Months")
ax[1].set_xlabel("Months")
ax[0].set_ylabel("Rainfall")
ax[1].set_ylabel("Rainfall")
plt.show()

# Task 10: Create a heatmap showing student performance across multiple subjects.
# Solution

# Making The Data
data = {
    "Maths": [78, 65, 90, 55],
    "Science": [85, 70, 92, 60],
    "English": [88, 72, 95, 58],
    "SST": [75, 80, 89, 65],
    "Computer": [90, 85, 96, 70],
}

# Loding The Data
df = pd.DataFrame(data, index=["Student1", "Student2", "Student3", "Student4"])

# Making The Heatmap
sns.heatmap(df, annot=True)
plt.show()

# Task 11: Build a box plot comparing exam scores from different classes.
# Solution

# Making The Dataset
class_a = [65, 70, 75, 80, 85, 90, 95]
class_b = [50, 55, 60, 65, 70, 75, 80]
class_c = [72, 74, 76, 78, 80, 82, 84]

# Making The Box Plot
bp = plt.boxplot([class_a, class_b, class_c])
plt.show()

# Task 13: Visualize website traffic trends for an entire year with moving averages.
# Solution

# Making The Dataset
days = np.arange(1, 366)
traffic = np.random.randint(1000, 5000, 365)

# Plotting The Lines
moving_avg = pd.Series(traffic).rolling(window=30).mean()
plt.figure(figsize=(12, 6))
plt.plot(days, traffic, alpha=0.3, label="Daily Traffic")
plt.plot(days, moving_avg, linewidth=3, label="30-Day Moving Average")

# Labeling
plt.title("Website Traffic Trend")
plt.xlabel("Days")
plt.ylabel("Visitors")

# Plotting The Graph
plt.legend()
plt.show()

# 3D VISUALIZATION TASKS 🚀
# Task 16: Create a 3D scatter plot representing student marks in Maths, Physics, and Chemistry.

# Solution

# Making The Dataset
students = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
maths = [78, 65, 90, 88, 72, 95, 81, 69, 84, 92]
physics = [70, 68, 92, 85, 75, 90, 80, 72, 82, 94]
chemistry = [75, 60, 95, 82, 78, 88, 85, 74, 86, 90]

# Plotting The Graph
fig = plt.figure()
ax = plt.subplot(projection="3d")
ax.scatter3D(maths, physics, chemistry, c="red")
plt.show()

# Task 17: Build a 3D paraboloid surface plot
# Solution

# Making The Data
X = np.linspace(-5, 5, 50)
Y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(X, Y)
Z = 100 - (X**2 + Y**2)

# Plotting The Graph
fig = plt.figure(figsize=(15, 10))
ax = plt.subplot(projection="3d")
p = ax.plot_surface(Z, Y, Z, cmap="magma")
fig.colorbar(p)
plt.show()
# Making The Data
# Task 18: Create a 3D wireframe plot of a mathematical surface.
# Solution

# Making The Data
X = np.linspace(-10, 5, 100)
Y = np.linspace(-10, 5, 100)
Z, Y = np.meshgrid(X, Y)

Z = 100 - (X**2 + Y**2)
fig = plt.figure(figsize=(15, 10))
ax = plt.subplot(projection="3d")
p = ax.plot_wireframe(X, Y, Z, color="pink")
fig.colorbar(p)
plt.show()

# Task 19: Visualize a mountain terrain using a 3D surface plot.
# Making The Data
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = (
    80 * np.exp(-((X - 3) ** 2 + (Y - 3) ** 2) / 10)
    + 120 * np.exp(-((X + 2) ** 2 + (Y + 1) ** 2) / 15)
    + 60 * np.exp(-((X - 5) ** 2 + (Y + 4) ** 2) / 8)
)

# Plotting The Graph
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8))
p = ax.plot_surface(
    X,
    Y,
    Z,
    cmap="magma",
)
plt.show()


# =========================================================================================== #
#                                     Code End -> 400+ LINES
# ============================================================================================ #
