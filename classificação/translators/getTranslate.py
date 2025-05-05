import numpy as np

def traduz_credit_data(input_vector):
    """
    Traduz um vetor de entrada do modelo de crédito para uma descrição humana.

    Args:
        input_vector (list ou np.array): Vetor de dados do cliente.

    Returns:
        dict: Dicionário com o significado de cada feature.
    """
    feature_names = {
        0: "Idade",
        1: "Renda Mensal",
        2: "Dívida Mensal",
        # Se você usar mais features, adicione aqui
    }

    # Confere se o input é array, se não for transforma
    if not isinstance(input_vector, (list, np.ndarray)):
        raise ValueError("A entrada deve ser uma lista ou um np.array.")

    if len(input_vector) != len(feature_names):
        raise ValueError(f"Entrada esperada com {len(feature_names)} elementos, mas recebeu {len(input_vector)}.")

    translated = {}
    for idx, value in enumerate(input_vector):
        translated[feature_names[idx]] = value

    return translated
