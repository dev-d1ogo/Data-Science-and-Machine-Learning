import numpy as np

from classificação.encoders.oneHotEncoder import oneHotEncoding
from classificação.data.getData import getData
from classificação.dataPreprocessing.clearData import fillInvalidValues
from classificação.dataPreprocessing.getForecasterAndClass import getForecasterAndClassOnBaseCredit, getForecasterAndClassOnBaseCensus
from classificação.dataPreprocessing.getNullData import getNullData
from classificação.encoders.labelEncoder import labelEncoding
from classificação.dataPreprocessing.preprocessingData import standarnisationData, standarnisationDataToNdArray
from classificação.training.getTrainingBase import getTrainingBaseData


import pickle
basecredit = getData('../data/credit_data.csv')

unique_values, counts = np.unique(basecredit['default'], return_counts=True)

for value, count in zip(unique_values, counts):
     print(f"{count} pessoas {"" if value == 0 else "não"} pagaram o emprestimo")

# Gera os graficos

# generateCreditChart(basecredit)

print(basecredit.describe())
basecredit = fillInvalidValues(basecredit)

nullRowsOnBaseCredit = getNullData(basecredit)

print(nullRowsOnBaseCredit)

'''
     Devemos escolher dados previsores e uma clase, onde:
      - Dados Previsores:
          São as informações de entrada usadas para fazer previsões (ex: idade, salário), onde normalmente são represeYtadYs por X.

      - Dado de Classe:
          É o que queremos prever (ex: se comprou ou não), onde normalmente representamos por Y.
'''
data = getForecasterAndClassOnBaseCredit(basecredit)

X_credit = data.X
Y_credit = data.Y

print(X_credit[:, 0].max(), X_credit[:, 1].max(), X_credit[:, 2].max())

data = standarnisationData(X_credit)

# X_credit padronizado

X_credit = data.X

print(X_credit[:, 0].max(), X_credit[:, 1].max(), X_credit[:, 2].max())


baseCensus = getData('../data/census.csv')

print(baseCensus.describe())

nullRowsOnBaseCensus = getNullData(baseCensus)

# Note que diferente da outra base essa aqui n tem valores falsy

print(nullRowsOnBaseCensus)

# Quantidade de colunas
print(f"Quantidade de colunas do census {len(baseCensus.columns)}")
# generateCensusChart(baseCensus)

censusData = getForecasterAndClassOnBaseCensus(baseCensus)

X_census = censusData.X
Y_census = censusData.Y

print(f"Dados previsores: {X_census[0]}")
print(f"Classificação: {Y_census[0]}")

# # Temos que passar por um encoder a nossa base ainda dado o problema de Modelos lidarem com dados matematicos e não classificatorios
# labelEncoding(X_census)
#
# print(f"Dados previsores: {X_census[0]}")

X_census = oneHotEncoding(X_census)

print(f"Dados previsores: {X_census[0]}")

# Escalonamento dos valores usando standarnisation para padrozinar os valores

X_census = standarnisationDataToNdArray(X_census)


print(f"Dados previsores: {X_census[0]}")

credit_data = getTrainingBaseData(X_credit, Y_credit, 0.25)

X_credit_train, X_credit_test, y_credit_train, y_credit_test = (
    credit_data["X_train"],
    credit_data["X_test"],
    credit_data["Y_train"],
    credit_data["Y_test"],
)

print(X_credit_train.shape)

census_data = getTrainingBaseData(X_census, Y_census, 0.15)

X_census_train, X_census_test, y_census_train, y_census_test = (
     census_data["X_train"],
     census_data["X_test"],
     census_data["Y_train"],
     census_data["Y_test"],
)

print(X_census_train.shape)

# Gera um arquivo para nos não termos que pre processar a base a toda hora
with open("../data/censusTraining.pkl", mode= "wb") as f:
    pickle.dump([X_census_train, y_census_train, X_census_test, y_census_test], f)

with open("../data/creditTraining.pkl", mode= "wb") as f:
    pickle.dump([X_credit_train, y_credit_train, X_credit_test, y_credit_test], f)

