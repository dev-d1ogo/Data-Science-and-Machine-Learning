import numpy as np
from sklearn.preprocessing import LabelEncoder

def labelEncoding(X_census: np.ndarray) -> None:
    # Criação dos LabelEncoders para colunas específicas
    label_encoder_workclass = LabelEncoder()
    label_encoder_education = LabelEncoder()
    label_encoder_marital = LabelEncoder()
    label_encoder_occupation = LabelEncoder()
    label_encoder_relationship = LabelEncoder()
    label_encoder_race = LabelEncoder()
    label_encoder_sex = LabelEncoder()
    label_encoder_country = LabelEncoder()

    # Encoding com nomes claros
    X_census[:, 1] = label_encoder_workclass.fit_transform(X_census[:, 1])
    X_census[:, 3] = label_encoder_education.fit_transform(X_census[:, 3])
    X_census[:, 5] = label_encoder_marital.fit_transform(X_census[:, 5])
    X_census[:, 6] = label_encoder_occupation.fit_transform(X_census[:, 6])
    X_census[:, 7] = label_encoder_relationship.fit_transform(X_census[:, 7])
    X_census[:, 8] = label_encoder_race.fit_transform(X_census[:, 8])
    X_census[:, 9] = label_encoder_sex.fit_transform(X_census[:, 9])
    X_census[:, 13] = label_encoder_country.fit_transform(X_census[:, 13])


def labelEncodingCreditRiskData(X_creditRisk: np.ndarray) -> None:
    # Criação dos LabelEncoders para colunas específicas
    label_encoder_history = LabelEncoder()
    label_encoder_debt = LabelEncoder()
    label_encoder_guarantees = LabelEncoder()
    label_encoder_income = LabelEncoder()




    # Encoding com nomes claros
    X_creditRisk[:, 0] = label_encoder_history.fit_transform(X_creditRisk[:, 0])
    X_creditRisk[:, 1] = label_encoder_debt.fit_transform(X_creditRisk[:, 1])
    X_creditRisk[:, 2] = label_encoder_guarantees.fit_transform(X_creditRisk[:, 2])
    X_creditRisk[:, 3] = label_encoder_income.fit_transform(X_creditRisk[:, 3])

    classesEncoder = {
        "history": label_encoder_history,
        "debt": label_encoder_debt,
        "guarantees": label_encoder_guarantees,
        "income": label_encoder_income
    }

    print("Conversão:")
    for nome_coluna, encoder in classesEncoder.items():
        print(f"\nColuna: {nome_coluna}")
        for i, valor_original in enumerate(encoder.classes_):
            print(f"{valor_original} → {i}")