from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
import numpy as np

@dataclass
class ScaledData:
    X: np.ndarray

def standarnisationData(X_data:np.ndarray) -> ScaledData:
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_data)

    return ScaledData(X_scaled)

def standarnisationDataToNdArray(X_census:np.ndarray) -> np.ndarray:
    scaler = StandardScaler()

    X_census = scaler.fit_transform(X_census)

    return X_census

