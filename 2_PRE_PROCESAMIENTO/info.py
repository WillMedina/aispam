import pandas as pd
df = pd.read_csv('../DATASETS_GENERADOS/dataset_spam_ham_flax-community_gpt-2-spanish_15000_48.csv')

print(df.head())
print("Shape:", df.shape)
print("\nHead:")
print(df.head())
print("\nTail:")
print(df.tail())
print("\nData types:")
print(df.dtypes)
print("\nDescriptive statistics:")
#print(df.describe(include='all', datetime_is_numeric=True))
print(df.describe())