import pandas as pd

FILE_PATH = "structured_analytics/data/raw/clinical_data.csv"

def main():
    df = pd.read_csv(FILE_PATH)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nShape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nSummary statistics:")
    print(df.describe(include="all"))

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print("\nCategorical columns:", categorical_cols)

    for col in categorical_cols:
        print(f"\nUnique values in '{col}':")
        print(df[col].value_counts(dropna=False))

    print("\nTarget distribution: mortality")
    print(df["mortality"].value_counts(dropna=False))
    print(df["mortality"].value_counts(normalize=True, dropna=False))

if __name__ == "__main__":
    main()