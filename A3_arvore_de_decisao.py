import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('focos_qmd_inpe_2019-01-01_2023-12-31_19.651277.csv', low_memory=False)

df['DataHora'] = pd.to_datetime(df['DataHora'], format='%Y/%m/%d %H:%M:%S', errors='coerce')
df['Ano'] = df['DataHora'].dt.year
df['Mes'] = df['DataHora'].dt.month


sns.countplot(data=df, x='Ano')
plt.title("Distribuição Anual de Queimadas")
plt.show()

# Remover linhas que não são numéricas na coluna 'Latitude' e 'Longitude'
df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

# Remover NaN
df = df.dropna(subset=['Latitude', 'Longitude', 'RiscoFogo'])

"""
sns.scatterplot(data=df, x='Longitude', y='Latitude', hue='RiscoFogo')
plt.title("Localização das Queimadas")
plt.show()
"""

# Codificando a coluna 'Bioma'
label_encoder = LabelEncoder()
df['Bioma'] = label_encoder.fit_transform(df['Bioma'])


# Selecionar colunas relevantes
features = ['DiaSemChuva', 'Precipitacao', 'Latitude', 'Longitude', 'Bioma']
target = 'RiscoFogo'  # Usando RiscoFogo como variável-alvo
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar o modelo
model = DecisionTreeClassifier(max_depth=5, random_state=42)

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliação do modelo
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))
print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred))

# Visualizar previsões de risco de queimada no mapa
df_test = X_test.copy()
df_test['Predito_RiscoFogo'] = y_pred

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_test, x='Longitude', y='Latitude', hue='Predito_RiscoFogo')
plt.title("Áreas de Risco de Queimada (Previsão)")
plt.show()

"""

print("Número de linhas:", data.shape[0])

# Pré-processamento dos dados
data['ano'] = pd.to_datetime(data['data_pas']).dt.year

# Agrupando dados por município e bioma
area_risk = data.groupby(['municipio', 'bioma']).size().reset_index(name='count')

# Encontrar áreas (cidades e biomas) com risco maior
top_areas = area_risk.sort_values(by='count', ascending=False).head(10)

# Plotar gráfico dos municípios com mais riscos
plt.figure(figsize=(12, 8))
sns.barplot(x='count', y='municipio', hue='bioma', data=top_areas, dodge=False)
plt.title("Top 10 Municípios com Maior Risco de Queimadas")
plt.xlabel("Número de Queimadas")
plt.ylabel("Município")
plt.show()

# Codificação das variáveis categóricas
encoder_municipio = LabelEncoder()
data['municipio_encoded'] = encoder_municipio.fit_transform(data['municipio'])

encoder_bioma = LabelEncoder()
data['bioma_encoded'] = encoder_bioma.fit_transform(data['bioma'])

# Agrupando e contando
data['count'] = data.groupby(['municipio', 'bioma'])['data_pas'].transform('count')
print(data)
limiar = 80

# Definindo o target
target = (data['count'] > limiar).astype(int)

print(target.value_counts())

# Seleção de características
features = data[['municipio_encoded', 'bioma_encoded', 'lat', 'lon']]

# Dividir o conjunto de dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Normalização dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinamento usando Redes Neurais
model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Predição
y_pred = model.predict(X_test_scaled)

# Avaliação
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Exibir as áreas prioritárias com maior risco
print("Municípios com maior tendência de queimadas:")
for area in top_areas.itertuples():
    print(f"Município: {area.municipio}, Bioma: {area.bioma}, Incident Count: {area.count}")

# Criando DataFrame com as previsões
predictions = model.predict(X_test_scaled)
prediction_df = pd.DataFrame(predictions, columns=['predictions'])
prediction_df['index'] = X_test.index  # Adiciona o índice do X_test

# Merge correto com o DataFrame original que contém o índice
data = data.reset_index()  # Reseta para adicionar índice original ao DataFrame data
data = data.merge(prediction_df[['index', 'predictions']], on='index', how='left')

areas_alto_risco = data[data['predictions'] == 1]
areas_baixo_risco = data[data['predictions'] == 0]

# Exibindo áreas de alto risco
#print(areas_alto_risco[['municipio', 'bioma', 'count']])

# Exibindo áreas de baixo risco
#print(areas_baixo_risco[['municipio', 'bioma', 'count']])

print("\n")

# Contando áreas de alto risco por município
print("Alto Risco Por municipio:")
alto_risco_por_municipio = areas_alto_risco.groupby('municipio').size().reset_index(name='alto_risco_count')
print(alto_risco_por_municipio)

# Contando áreas de baixo risco por município
print("Baixo Risco Por municipio:")
baixo_risco_por_municipio = areas_baixo_risco.groupby('municipio').size().reset_index(name='baixo_risco_count')
print(baixo_risco_por_municipio)

# Identificar os 10 municípios com maior risco
maior_risco = areas_alto_risco.groupby('municipio').size().reset_index(name='count').sort_values(by='count', ascending=False).head(10)

# Identificar os 10 municípios com menor risco
menor_risco = areas_baixo_risco.groupby('municipio').size().reset_index(name='count').sort_values(by='count', ascending=True).head(10)

# Concatenar os dois DataFrames para plotar
maior_risco['Risco'] = 'Alto'
menor_risco['Risco'] = 'Baixo'

risco_combined = pd.concat([maior_risco, menor_risco])

# Plotar gráfico
plt.figure(figsize=(14, 8))
sns.barplot(x='count', y='municipio', hue='Risco', data=risco_combined, dodge=True)
plt.title("Top 10 Municípios com Maior e Menor Risco de Queimadas")
plt.xlabel("Número de Queimadas")
plt.ylabel("Município")
plt.legend(title='Risco')
plt.show()

"""