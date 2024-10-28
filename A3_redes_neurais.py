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

label_encoder = LabelEncoder()
df['Bioma'] = label_encoder.fit_transform(df['Bioma'])