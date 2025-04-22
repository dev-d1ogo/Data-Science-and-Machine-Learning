from sklearn.model_selection import train_test_split
from typing import TypedDict
import numpy as np

class TrainingBase(TypedDict):
    X_train: np.ndarray
    X_test: np.ndarray
    Y_train: np.ndarray
    Y_test: np.ndarray

def getTrainingBaseData(X_data: np.ndarray, Y_data: np.ndarray, testSize: float) -> TrainingBase:
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_data, Y_data,
        test_size=testSize,  # 25% para validação
        random_state=42,  # para garantir que a divisão sempre será igual
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "Y_train": Y_train,
        "Y_test": Y_test
    }
