import pandas as pd

# Make fake Excel
df = pd.DataFrame({"Name": ["Apple", "Banana"], "Code": [1, 2]})

print("=== TABLE WAY (OLD) ===")
print(df.iloc[0])  # Table row
# Name    Apple
# Code        1
# Name: 0, dtype: object

print("\n=== LIST WAY (NEW - NO TABLE!) ===")
data_list = df.to_dict()
print(data_list[0])  # List item
# {'Name': 'Apple', 'Code': 1}