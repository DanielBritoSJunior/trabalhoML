import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. CARREGAR DADOS
df = pd.read_csv('Chicago-RE_HousePrice.csv')

# 2. LIMPEZA (Tratamento de Nulos e Outliers)
# Remove linhas onde o preço (alvo) está vazio
df = df.dropna(subset=['Price']) 
# Preenche buracos em outras colunas com a mediana 
imputer = SimpleImputer(strategy='median')
df_clean = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# 3. ANÁLISE EXPLORATÓRIA (Gera os gráficos para o relatório)
# --- ADICIONE ISSO NA PARTE DE ANÁLISE EXPLORATÓRIA (EDA) ---

# 1. Distribuição do Preço (Exigência do item 4 do roteiro)
plt.figure(figsize=(10, 6))
sns.histplot(df_clean['Price'], kde=True, color='blue')
plt.title('Distribuição da Variável Alvo (Preço em Chicago)')
plt.xlabel('Preço')
plt.ylabel('Frequência')
plt.savefig('distribuicao_preco.png') # Salva o gráfico solicitado

# 2. Boxplot de Outliers (Exigência do item 4 do roteiro)
plt.figure(figsize=(10, 6))
sns.boxplot(x=df_clean['Price'])
plt.title('Identificação de Outliers no Preço')
plt.savefig('boxplot_outliers.png') # Salva para mostrar os valores fora da curva

# 3. Heatmap de Correlação (Você já tem, mas mantenha para o relatório)
plt.figure(figsize=(10, 8))
sns.heatmap(df_clean.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap de Correlação')
plt.savefig('heatmap_chicago.png')

# 4. PRÉ-PROCESSAMENTO [cite: 16]
X = df_clean.drop('Price', axis=1)
y = df_clean['Price']

# Divisão Treino e Teste (80/20) [cite: 19]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ESCALONAMENTO (Obrigatório para SVR e Linear) 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. TREINAMENTO DOS 3 MODELOS [cite: 21, 22]
models = {
    "Regressão Linear": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(kernel='rbf')
}

def calcular_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Loop para treinar e avaliar
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    
    print(f"\n--- {name} ---")
    print(f"R2: {r2_score(y_test, preds):.4f}") # [cite: 29]
    print(f"MAE: {mean_absolute_error(y_test, preds):.4f}") # [cite: 30]
    print(f"MAPE: {calcular_mape(y_test, preds):.2f}%") # [cite: 31]
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.4f}") # [cite: 32]

# Pegando os palpites do melhor modelo (Random Forest)
y_pred_rf = models["Random Forest"].predict(X_test_scaled)

plt.figure(figsize=(8, 8))
# Desenha os pontos (Real no eixo X, Chute da IA no eixo Y)
plt.scatter(y_test, y_pred_rf, alpha=0.5, color='blue', label='Previsões')

# Desenha a linha ideal de 45 graus (Onde o Real é igual ao Predito)
limit_min = min(y_test.min(), y_pred_rf.min())
limit_max = max(y_test.max(), y_pred_rf.max())
plt.plot([limit_min, limit_max], [limit_min, limit_max], color='red', linestyle='--', label='Perfeição (45°)')

plt.title('Valores Reais vs. Preditos (Chicago - Random Forest)')
plt.xlabel('Preço Real (Gabarito)')
plt.ylabel('Preço Predito (Palpite da IA)')
plt.legend()
plt.grid(True)
plt.savefig('real_vs_predito_chicago.png') # Salva o arquivo para o seu relatório