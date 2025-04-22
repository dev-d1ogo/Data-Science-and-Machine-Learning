import pandas as pd

from classificação.data.getData import getData
from classificação.dataPreprocessing.getForecasterAndClass import getForecasterAndClassesOnCreditRisk
from classificação.encoders.labelEncoder import labelEncodingCreditRiskData
import pickle

from sklearn.naive_bayes import GaussianNB

baseCreditRisk = getData('../../data/risco_credito.csv')

creditRiskData = getForecasterAndClassesOnCreditRisk(baseCreditRisk)

X_riskCredit = creditRiskData.X
Y_riskCredit = creditRiskData.Y

print(f"Base de dados antes do processamento: ${X_riskCredit}")

# Pre processamento dos dados
labelEncodingCreditRiskData(X_riskCredit)

print(f"Base de dados depois do processamento: ${X_riskCredit}")


with open("../../data/risckCredit.pkl", mode= "wb") as f:
    pickle.dump([X_riskCredit, Y_riskCredit], f)


# Vamos treinar o algoritmo criando a nossa tabela

naiveBayesCreditRisk = GaussianNB()

naiveBayesCreditRisk.fit(X_riskCredit, Y_riskCredit)

# Fazendo uma previsão

'''
Primeira previsão: 
    historia: boa(0)
    divida: alta (0)
    garantia: nenhuma (1)
    renda: >35 k (2)
Segunda previsão: 
    historia: ruim(2)
    divida: alta (0)
    garantia: adequada (0)
    renda: <15 k (0)
'''

firstPrediction = [0,0,1,2]
secondPrediction = [2,0,0,0]

predition = naiveBayesCreditRisk.predict([firstPrediction,secondPrediction ])

print(f"A previsão para um cliente com esse histórico é: \n"
      f"- História: boa (0)\n"
      f"- Divida: alta (0)\n"
      f"- Garantia: nenhuma (1)\n"
      f"- Renda: >35 k (2) é: {predition[0].upper()}\n")

print(f"A previsão para um cliente com esse histórico é: \n"
      f"- História: ruim (0)\n"
      f"- Divida: alta (0)\n"
      f"- Garantia: adequada (1)\n"
      f"- Renda: >15k (0) => é: {predition[1].upper()}")