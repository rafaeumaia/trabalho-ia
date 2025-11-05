import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ==============================================
# 1. Carregar o dataset CIFAR-10
# ==============================================
print("Carregando o dataset CIFAR-10...")
(XTreino, yTreino), (XTeste, yTeste) = keras.datasets.cifar10.load_data()

# ==============================================
# 2. Preparar os dados
# ==============================================
XTreino_reshaped = XTreino.reshape((XTreino.shape[0], -1))
XTeste_reshaped = XTeste.reshape((XTeste.shape[0], -1))

XTreino_normalized = XTreino_reshaped / 255.0
XTeste_normalized = XTeste_reshaped / 255.0

yTreino_flat = yTreino.flatten()
yTeste_flat = yTeste.flatten()

# ==============================================
# 3. Definir classificadores
# ==============================================
classificadores = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# ==============================================
# 4. Criar pasta para matrizes
# ==============================================
os.makedirs("imagens", exist_ok=True)

# ==============================================
# 5. Treinar e avaliar
# ==============================================
resultados_gerais = []

for nome, modelo in classificadores.items():
    print(f"\n===============================")
    print(f"Treinando e avaliando: {nome}")
    print(f"===============================\n")

    modelo.fit(XTreino_normalized, yTreino_flat)
    y_pred = modelo.predict(XTeste_normalized)

    # Relatório detalhado (por classe + médias)
    report = classification_report(yTeste_flat, y_pred, digits=2)
    print(report)

    # Métricas gerais para resumo final
    acc = accuracy_score(yTeste_flat, y_pred)
    prec = precision_score(yTeste_flat, y_pred, average='macro')
    rec = recall_score(yTeste_flat, y_pred, average='macro')
    f1 = f1_score(yTeste_flat, y_pred, average='macro')
    resultados_gerais.append([nome, acc, prec, rec, f1])

    # Matriz de confusão geral
    cm = confusion_matrix(yTeste_flat, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title(f"Matriz de Confusão - {nome}")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    caminho_img = f"imagens/matriz_{nome.replace(' ', '_')}.png"
    plt.savefig(caminho_img)
    plt.close()
    print(f"✅ Matriz de confusão salva em: {caminho_img}\n")

# ==============================================
# 6. Resumo final geral
# ==============================================
print("===== RESULTADOS FINAIS GERAIS =====")
print(f"{'Classificador':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
for nome, acc, prec, rec, f1 in resultados_gerais:
    print(f"{nome:<20} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f}")
