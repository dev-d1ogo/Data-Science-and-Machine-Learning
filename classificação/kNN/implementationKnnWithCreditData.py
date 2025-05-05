import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from yellowbrick.classifier import ConfusionMatrix

from classificação.dataPreprocessing.preprocessingData import standarnisationData
from classificação.translators.getTranslate import traduz_credit_data

# Carregando os dados pre processados
# Note que os dados salvos ja passaram por um processo de stan
with open("../../data/creditTraining.pkl", "rb") as f:
    X_credit_train, y_credit_train, X_credit_test, y_credit_test = pickle.load(f)

# Calcula o valor da distaincia euclediana entre os pontos

knn_credit = KNeighborsClassifier(n_neighbors=12, metric='minkowski', p=2, weights='uniform')

knn_credit.fit(X_credit_train, y_credit_train)

predictions = knn_credit.predict(X_credit_test)

print("Previsoes de treinamento: ", predictions)

print("Previsoes de fato: ", y_credit_test)

accuracy = accuracy_score(y_credit_test, predictions)

print("Accuracy: ", accuracy)


# Treina um novo scaler baseado nos dados de treino
scaler = StandardScaler()
scaler.fit(X_credit_train)

# Exemplo: idade=30, renda=4000, divida=500
input_data = np.array([[30, 4000, 500]])

# Padroniza o input igual ao treinamento
input_data_scaled = standarnisationData(input_data)

print(input_data_scaled)


prediction = knn_credit.predict(input_data_scaled.X)

# Interpreta o resultado
if prediction[0] == 0:
    print("Previsão: A pessoa **paga** o empréstimo.")
else:
    print("Previsão: A pessoa **NÃO paga** o empréstimo.")

cm = ConfusionMatrix(knn_credit)

cm.fit(X_credit_train, y_credit_train)
cm.score(X_credit_test, y_credit_test)

cm.show()