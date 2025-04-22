from sklearn.tree import DecisionTreeClassifier
import pickle


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