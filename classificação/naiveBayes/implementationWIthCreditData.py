import pickle

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
import numpy as np
from yellowbrick.classifier import ConfusionMatrix

# Carregando os dados pre processados
with open("../../data/creditTraining.pkl", "rb") as f:
    X_credit_train, y_credit_train, X_credit_test, y_credit_test = pickle.load(f)



print(X_credit_train[:5])

naiveBayesCredit_data = GaussianNB()

# Treinamos o algoritmo para ele criar a tabela de probabilidades com os dados de treinamento
naiveBayesCredit_data.fit(X_credit_train, y_credit_train)  # Usando y_credit_train, não y_credit_test

# Em seguida passamos os dados de teste para o algoritmo já treinado
predictions = naiveBayesCredit_data.predict(X_credit_test)

realData = y_credit_test


accuracy = accuracy_score(realData, predictions)

print(accuracy)

# Matriz de Confusão:
# Compara os valores reais com os valores previstos pelo modelo.
# Classe 0 = "Paga o crédito"
# Classe 1 = "Não paga o crédito"

#          Previsto
#             0     1
# Real  0  [415,    7]   -> 415 pagadores corretamente identificados (VP)
#                         -> 7 pagadores classificados como inadimplentes (FN)
#       1  [32,    46]   -> 32 inadimplentes classificados como pagadores (FP)
#                         -> 46 inadimplentes corretamente identificados (VN)

# Interpretação:
# - VP (Verdadeiro Positivo): modelo acertou quem paga (classe 0)
# - VN (Verdadeiro Negativo): modelo acertou quem não paga (classe 1)
# - FP (Falso Positivo): modelo disse que pagaria, mas não pagou
# - FN (Falso Negativo): modelo disse que não pagaria, mas pagou

accuracyMatrix = confusion_matrix(realData, predictions)

print(accuracyMatrix)


cm = ConfusionMatrix(naiveBayesCredit_data)

cm.fit(X_credit_train, y_credit_train)
cm.score(X_credit_test, y_credit_test)

cm.show()