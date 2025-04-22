import pandas as pd

def getData(path: str) -> pd.DataFrame:
    base_credit = pd.read_csv(path)

    return base_credit