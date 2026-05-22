import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("orders.csv")
df["OrderDate"] = pd.to_datetime(df["OrderDate"])
df["TotalAmount"] = df["Quantity"] * df["Price"]

print(df)
print(df["TotalAmount"].sum())
print(df["TotalAmount"].mean())
print(df["Customer"].value_counts())
print(df[df["TotalAmount"] > 500])
print(df.sort_values(by="OrderDate", ascending=False))
print(df[(df["OrderDate"] >= "2023-06-05") & (df["OrderDate"] <= "2023-06-10")])
print(df.groupby("Category")[["Quantity", "TotalAmount"]].sum())
print(df.groupby("Customer")["TotalAmount"].sum().sort_values(ascending=False).head(3))

orders_by_date = df.groupby("OrderDate").size()
orders_by_date.index = orders_by_date.index.strftime("%Y-%m-%d")
orders_by_date.plot(kind="bar")
plt.show()

df.groupby("Category")["TotalAmount"].sum().plot(kind="pie", autopct="%1.1f%%")
plt.ylabel("")
plt.show()
