import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from narwhals.stable.v1 import DataFrame


def generateCreditChart(basecredit:DataFrame) -> None:
    sns.countplot(x=basecredit['default'], hue=basecredit['default'], palette={0: "green", 1: "red"})

    plt.title("Distribuição de Pagamento de Empréstimos")
    plt.xlabel("Default (0 = pagou, 1 = não pagou)")
    plt.ylabel("Quantidade de Pessoas")
    plt.show()

    graph = px.scatter_matrix(basecredit, dimensions=["age", "income"], )
    graph.show()


def generateCensusChart(baseCensus:DataFrame) -> None:
    sns.countplot(x=baseCensus['income'], hue=baseCensus['income'], palette={" <=50K": "blue", " >50K": "green"})

    plt.title("Distribuição de renda da população")
    plt.xlabel("Income (<=50K = classe média , >50K = classe alta)")
    plt.ylabel("Quantidade de Pessoas")
    plt.show()

    # graph = px.scatter_matrix(baseCensus, dimensions=["age", "income"], )
    # graph.show()
