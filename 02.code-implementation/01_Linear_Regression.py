import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('datasets/Ecommerce Customers')
print(df.head()) # Display the first few rows of the dataframe

df.info() # Get information about the dataframe
print(df.describe()) # Get statistical summary of the dataframe




