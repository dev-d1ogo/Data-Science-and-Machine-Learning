import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

def transformEncondingNdArrayToDataFrame(X: np.ndarray, colunas_originais: list[str], indices_categoricos: list[int]):
    """
    Aplica LabelEncoder e OneHotEncoder às colunas categóricas e retorna um DataFrame com os nomes de colunas preservados.

    Parâmetros:
    - X: np.ndarray com os dados brutos.
    - colunas_originais: nomes originais das colunas.
    - indices_categoricos: índices das colunas categóricas.

    Retorno:
    - DataFrame final com colunas bem nomeadas.
    """
    X_copy = X.copy()

    # Aplicando LabelEncoder nas colunas categóricas
    for idx in indices_categoricos:
        le = LabelEncoder()
        X_copy[:, idx] = le.fit_transform(X_copy[:, idx])

    # Nome das colunas categóricas
    colunas_categoricas = [colunas_originais[i] for i in indices_categoricos]

    # Aplicando OneHotEncoder
    onehot = ColumnTransformer(
        transformers=[("OneHot", OneHotEncoder(sparse_output=False), indices_categoricos)],
        remainder="passthrough"
    )

    X_encoded = onehot.fit_transform(X_copy)

    # Recuperando os nomes das colunas one-hot
    onehot_feature_names = onehot.named_transformers_['OneHot'].get_feature_names_out(colunas_categoricas)

    # Pegando colunas não categóricas
    colunas_numericas = [col for i, col in enumerate(colunas_originais) if i not in indices_categoricos]

    # Juntando todos os nomes das colunas no mesmo DataFrame
    todas_colunas = list(onehot_feature_names) + colunas_numericas

    # Criando o DataFrame final
    df_final = pd.DataFrame(X_encoded, columns=todas_colunas)

    df_final.to_csv("/home/diogo/Dev/personal/IA/MachineLearning/data/chama.csv", index=False)

