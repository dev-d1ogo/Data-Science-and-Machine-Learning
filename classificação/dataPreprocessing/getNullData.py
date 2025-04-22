import pandas as pd

def getNullData(baseDataSet: pd.DataFrame) -> pd.DataFrame.sum:
    return baseDataSet[baseDataSet.isnull().any(axis="columns")].sum()