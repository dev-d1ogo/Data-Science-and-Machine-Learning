from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class ForecasterData:
    X: np.ndarray
    Y: np.ndarray

def getForecasterAndClassOnBaseCredit(basecredit: pd.DataFrame) -> ForecasterData:
    X_credit = basecredit.iloc[:, 1:4].values
    Y_credit = basecredit.iloc[:, 4].values
    return ForecasterData(X=X_credit, Y=Y_credit)

def getForecasterAndClassOnBaseCensus(basecredit: pd.DataFrame) -> ForecasterData:
    X_census = basecredit.iloc[:, 0:14].values
    print(X_census)

    Y_census = basecredit.iloc[:, 14].values
    print(Y_census)
    return ForecasterData(X=X_census, Y=Y_census)

def getForecasterAndClassesOnCreditRisk(basecredit: pd.DataFrame) -> ForecasterData:
    X_riskCredit = basecredit.iloc[:, 0:4].values
    Y_riskCredit = basecredit.iloc[:, 4].values

    return ForecasterData(X=X_riskCredit, Y=Y_riskCredit)