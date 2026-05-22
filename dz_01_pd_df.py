import pandas as pd
import matplotlib.pyplot as plt


data = {
    "OrderID": [1001, 1002, 1003],
    "Customer": ["Alice", "Bob", "Alice"],
    "Product": ["Laptop", "Chair", "Mouse"],
    "Category": ["Electronics", "Furniture", "Electronics"],
    "Quantity": [1, 2, 3],
    "Price": [1500, 180, 25],
    "OrderDate": ["2023-06-01", "2023-06-03", "2023-06-05"]
}

df = pd.DataFrame(data)

# Конвертація дати
df["OrderDate"] = pd.to_datetime(df["OrderDate"])

# Новий стовпець
df["TotalAmount"] = df["Quantity"] * df["Price"]

print("Початкова таблиця:")
print(df)

# 3a. Сумарний дохід
total_income = df["TotalAmount"].sum()
print("\nСумарний дохід:", total_income)

# 3b. Середнє значення
avg_total = df["TotalAmount"].mean()
print("Середній чек:", avg_total)

# 3c. Кількість замовлень по клієнтах
orders_per_customer = df["Customer"].value_counts()
print("\nКількість замовлень:\n", orders_per_customer)

# 4. Замовлення > 500
print("\nЗамовлення > 500:")
print(df[df["TotalAmount"] > 500])

# 5. Сортування по даті
print("\nСортування по даті:")
print(df.sort_values(by="OrderDate", ascending=False))

# 6. Замовлення з 5 по 10 червня
filtered_dates = df[(df["OrderDate"] >= "2023-06-05") & (df["OrderDate"] <= "2023-06-10")]
print("\nЗамовлення 5-10 червня:")
print(filtered_dates)

# 7. Групування по категоріях
category_stats = df.groupby("Category").agg({
    "Quantity": "sum",
    "TotalAmount": "sum"
})
print("\nСтатистика по категоріях:")
print(category_stats)

# 8. ТОП-3 клієнтів
top_customers = df.groupby("Customer")["TotalAmount"].sum().sort_values(ascending=False).head(3)
print("\nТОП-3 клієнтів:")
print(top_customers)

# Графік кількості замовлень по датах
orders_by_date = df.groupby("OrderDate").size()
orders_by_date.index = orders_by_date.index.strftime("%Y-%m-%d")
orders_by_date.plot(kind="bar", color="skyblue")
plt.title("Кількість замовлень по датах")
plt.xlabel("Дата")
plt.ylabel("Кількість замовлень")
plt.tight_layout()
plt.show()

# Діаграма доходів по категоріях
income_by_category = df.groupby("Category")["TotalAmount"].sum()
income_by_category.plot(kind="pie", autopct="%1.1f%%")
plt.title("Розподіл доходів по категоріях")
plt.ylabel("")
plt.tight_layout()
plt.show()
