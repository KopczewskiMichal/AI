import pandas as pd
import warnings
import random
warnings.filterwarnings("ignore")


# Liczenie zepsutych danych
missing_values = ["n/a", "na", "--", "-", "NA"]
# df = pd.read_csv("better_iris.csv", na_values=missing_values)
df = pd.read_csv("iris_with_errors.csv", na_values=missing_values)


numeric_columns = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')


smaller_than_0 = df[(df[numeric_columns] < 0).any(axis=1)]
greater_than_max = df[df['sepal.length'] > 15]
contains_na = df[df.isna().any(axis=1)]

print(pd.concat([contains_na, smaller_than_0, greater_than_max]))

# Ulepszanie danych numerycznych
for col in numeric_columns:
  avg = round(df[col].mean(), 1)
  df[col].fillna(avg, inplace=True)
  df[col] = df[col].clip(lower=0, upper=15)

# Zmiana danych string
correct_strings = ["Virginica","Setosa","Versicolor"]
def replace_value(x):
    if x in correct_strings:
        return x
    else:
        return random.choice(correct_strings)

df['variety'] = df['variety'].apply(replace_value)


df.to_csv("better_iris.csv", index=False)

  





