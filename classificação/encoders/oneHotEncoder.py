import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def oneHotEncoding(X_census: np.ndarray) -> np.ndarray:
    # Criação dos OneHotEncoders para colunas

    oneHotEncoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(sparse_output=False), [1,3,5,6,7,8,9,13])], remainder="passthrough")

    X_census = oneHotEncoder.fit_transform(X_census)

    return X_census
