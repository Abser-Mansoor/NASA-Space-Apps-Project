import pandas as pd

df = pd.read_csv('meteomatics_data.csv', sep=";")

df.info()

print(df.head())
print(df.describe())
