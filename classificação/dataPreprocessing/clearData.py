import pandas as pd


def dropInvalidColumns(base_credit: pd.DataFrame) -> pd.DataFrame:
    cleaned_basecredit = base_credit.copy()

    invalid_mask = (
        (cleaned_basecredit["age"] < 0) |
        (cleaned_basecredit["age"].isnull()) |
        (cleaned_basecredit["age"] == False)
    )

    # Obtém os índices das linhas inválidas
    invalid_indexes = cleaned_basecredit[invalid_mask].index

    cleaned_basecredit = cleaned_basecredit.drop(invalid_indexes)

    print(cleaned_basecredit.describe())

    return cleaned_basecredit

def fillInvalidValues(base_credit: pd.DataFrame) -> pd.DataFrame:
    cleanedDataSet = dropInvalidColumns(base_credit)

    ageMean = cleanedDataSet["age"].mean()

    # Preenche valores negativos, nulos ou False com a média
    base_credit.loc[
        (base_credit["age"] < 0) |
        (base_credit["age"].isnull()) |
        (base_credit["age"] == False),
        "age"
    ] = ageMean

    return base_credit

