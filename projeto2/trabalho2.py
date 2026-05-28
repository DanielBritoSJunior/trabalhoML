import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Importando os 6 classificadores exigidos no roteiro do professor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# 1. CARREGAR DADOS (Dataset de Câncer de Mama)
df = pd.read_csv(r"projeto2\data.csv")

# CORREÇÃO: Limpa aspas ocultas e espaços vazios dos nomes das colunas
df.columns = df.columns.str.replace('"', '').str.strip()

# 2. LIMPEZA (Ajustes específicos para este dataset)
# Remove a coluna 'id' e a coluna fantasma 'Unnamed: 32' se existirem
df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

# Mapeando o Target de texto para número: M (Maligno) = 1, B (Benigno) = 0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# 3. ANÁLISE EXPLORATÓRIA (EDA exigida no relatório)
# Gráfico 1: Equilíbrio de classes
plt.figure(figsize=(6, 4))
sns.countplot(x='diagnosis', data=df, palette='Set2')
plt.title('Distribuição dos Diagnósticos (0 = Benigno, 1 = Maligno)')
plt.xticks([0, 1], ['Benigno (B)', 'Maligno (M)'])
plt.savefig('equilibrio_classes.png', dpi=300, bbox_inches='tight')
plt.close()

# Gráfico 2: Heatmap de Correlação (das primeiras 10 variáveis para não poluir)
plt.figure(figsize=(10, 8))
features_mean = [col for col in df.columns if '_mean' in col] + ['diagnosis']
sns.heatmap(df[features_mean].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap de Correlação (Características Médias)')
plt.savefig('heatmap_cancer.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. PRÉ-PROCESSAMENTO
# Com o cabeçalho limpo, o drop vai funcionar perfeitamente sem dar KeyError
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Divisão Treino e Teste (80/20) com stratify=y para garantir a mesma proporção de M e B em ambos os lados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ESCALONAMENTO (Crucial para KNN, SVM e Regressão Logística)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. TREINAMENTO DOS 6 MODELOS (Conforme exigido pelo professor)
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Regressão Logística": LogisticRegression(random_state=42),
    "Árvore de Decisão": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', random_state=42),
    "Naive Bayes": GaussianNB()
}

# Dicionário para guardar os resultados e comparar depois
resultados = {}

# Loop para treinar e avaliar todos os modelos de classificação
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    
    # Calculando as métricas de classificação
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    # Salvando no dicionário
    resultados[name] = [acc, prec, rec, f1]
    
    print(f"\n--- {name} ---")
    print(f"Acurácia:  {acc:.4f}")
    print(f"Precisão:  {prec:.4f}")
    print(f"Recall:    {rec:.4f}  <-- (Super importante: Evita Falso Negativo!)")
    print(f"F1-Score:  {f1:.4f}")

# 6. COMPARAÇÃO VISUAL (Tabela de Resultados)
df_resultados = pd.DataFrame.from_dict(resultados, orient='index',
                                       columns=['Acurácia', 'Precisão', 'Recall', 'F1-Score'])
print("\n=== TABELA COMPARATIVA FINAL ===")
print(df_resultados)

# 7. GERANDO A MATRIZ DE CONFUSÃO (Para o relatório)
# Escolha do modelo para a matriz de confusão (ex: Random Forest)
melhor_modelo_nome = "Random Forest"
melhor_modelo = models[melhor_modelo_nome]
preds_melhor = melhor_modelo.predict(X_test_scaled)
cm = confusion_matrix(y_test, preds_melhor)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Benigno', 'Maligno'], 
            yticklabels=['Benigno', 'Maligno'])
plt.title(f'Matriz de Confusão - {melhor_modelo_nome}')
plt.xlabel('Predito pela IA')
plt.ylabel('Realidade (Gabarito)')
plt.savefig('matriz_confusao.png', dpi=300, bbox_inches='tight')
plt.show()