# Dynamic Pricing Project (Ride-Sharing Dataset)
# Project Name: dynamic_pricing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# 1. Load Data
# -----------------------------
df = pd.read_csv("dynamic_pricing.csv")   # apna CSV file ka naam daalna

# -----------------------------
# 2. Data Cleaning
# -----------------------------
df = df.dropna()   # missing values remove

# Outlier removal (example on Historical_Cost_of_Ride)
Q1 = df['Historical_Cost_of_Ride'].quantile(0.25)
Q3 = df['Historical_Cost_of_Ride'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Historical_Cost_of_Ride'] >= Q1 - 1.5*IQR) & 
        (df['Historical_Cost_of_Ride'] <= Q3 + 1.5*IQR)]

# -----------------------------
# 3. EDA
# -----------------------------
plt.scatter(df['Historical_Cost_of_Ride'], df['Number_of_Riders'])
plt.xlabel("Historical Cost of Ride")
plt.ylabel("Number of Riders")
plt.title("Cost vs Riders (Demand)")
plt.show()

# -----------------------------
# 4. Feature Engineering
# -----------------------------
df['cost_bucket'] = pd.cut(df['Historical_Cost_of_Ride'], 
                           bins=[0,100,200,400,800], 
                           labels=['Low','Medium','High','Premium'])
df['loyalty_flag'] = np.where(df['Customer_Loyalty_Status']=="Yes",1,0)

# -----------------------------
# 5. ML Models
# -----------------------------
X = df[['Historical_Cost_of_Ride','Number_of_Drivers','Average_Ratings']]
y = df['Number_of_Riders']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train,y_train)

# Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train,y_train)

# -----------------------------
# 6. Optimization Logic
# -----------------------------
cost_per_ride = 50   # example fixed cost
prices = np.arange(50,800,20)
profits = []

for p in prices:
    demand_pred = rf.predict([[p, df['Number_of_Drivers'].mean(), df['Average_Ratings'].mean()]])[0]
    profit = (p * demand_pred) - cost_per_ride
    profits.append(profit)

optimal_price = prices[np.argmax(profits)]
max_profit = max(profits)

print("Optimal Ride Price:", optimal_price)
print("Maximum Profit:", max_profit)

# -----------------------------
# 7. Visualization
# -----------------------------
plt.plot(prices, profits, label="Profit Curve")
plt.axvline(optimal_price, color='red', linestyle='--', label=f"Optimal Price: {optimal_price}")
plt.xlabel("Ride Price")
plt.ylabel("Profit")
plt.title("Price vs Profit Curve")
plt.legend()
plt.show()

# -----------------------------
# 8. Business Insights
# -----------------------------
print("Insights:")
print("- Underpriced rides: high demand but low profit.")
print("- Overpriced rides: low demand, declining profit.")
print("- Loyalty customers respond better to optimized pricing.")
