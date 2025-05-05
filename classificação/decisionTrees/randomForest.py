"""
O algoritmo de random forest junta várias ávores de decisão
 - Gerar uma floresta inteira ao inves de apenas uma para tomar a decisão
 - Ou seja, varios algoritmos juntos para construir um mais "forte"
 - OBS: Usa a média quando queremeos uma regressão (valor numero) e votos da maioria para uma classificação

 Agora o random vem que as florestas são geradas com base nos atributos, onde K = numero de atributos selecionados.
 Ou seja, cada arvore vai ser gerada usando atributos diferentes.

 Ex: Nossa base de risco de credito tem 4 atributos sendo K = 3 ela vai gerar N arvores com aleatorização desses atributos
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
from yellowbrick.classifier import ConfusionMatrix

import matplotlib.pyplot as plt


with open("../../data/creditTraining.pkl", mode= "rb") as f:
    [X_credit_train, y_credit_train, X_credit_test, y_credit_test] = pickle.load(f)

print(X_credit_train.shape)
print(y_credit_train.shape)

random_forest_credit = RandomForestClassifier(n_estimators=100,criterion="entropy", random_state=0)

random_forest_credit.fit(X_credit_train, y_credit_train)

predicts = random_forest_credit.predict(X_credit_test)

print(f"Essas são as previsões: {predicts}")
print(f"Essas são os dados reais: {y_credit_test}")

accuracy = accuracy_score(y_credit_test, predicts)

print(f"Nosso algoritmo teve uma precisão de: {accuracy}")

cm = ConfusionMatrix(random_forest_credit)

cm.fit(X_credit_train, y_credit_train)
cm.score(X_credit_test, y_credit_test)

cm.show()