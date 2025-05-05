from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pickle

import matplotlib.pyplot as plt


with open("../../data/risckCredit.pkl", mode= "rb") as f:
    [X_riskCredit, Y_riskCredit] = pickle.load(f)

print(X_riskCredit[:5])
print(Y_riskCredit[:5])

# Criando a nossa arvore de decisão que pode ser configurada entropy para pegar o calculo da entropia, além de opcoes como o maximo de profundidade da arvore
decisionTreeCreditRisk = DecisionTreeClassifier(criterion="entropy", max_depth=5)


# Entramos com os dados previsores e a classe
decisionTreeCreditRisk.fit(X_riskCredit, Y_riskCredit)

score = decisionTreeCreditRisk.score(X_riskCredit, Y_riskCredit)

print(score)

importance = decisionTreeCreditRisk.feature_importances_

print(importance)

forecasters = ['history', 'debt', 'guarantees', 'income']
figure, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

tree.plot_tree(decisionTreeCreditRisk, feature_names=forecasters, class_names=decisionTreeCreditRisk.classes_)

plt.show()

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

predictions = decisionTreeCreditRisk.predict([firstPrediction, secondPrediction ])

print(f"A previsão para um cliente com esse histórico é: \n"
      f"- História: boa (0)\n"
      f"- Divida: alta (0)\n"
      f"- Garantia: nenhuma (1)\n"
      f"- Renda: >35 k (2) é: {predictions[0].upper()}\n")

print(f"A previsão para um cliente com esse histórico é: \n"
      f"- História: ruim (0)\n"
      f"- Divida: alta (0)\n"
      f"- Garantia: adequada (1)\n"
      f"- Renda: >15k (0) => é: {predictions[1].upper()}")
