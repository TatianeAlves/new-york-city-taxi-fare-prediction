# Importando bibliotecas
# Leitura e manipulação de dados 
import numpy as np
import pandas as pd
import kaggle
import math
import holidays
import time
from datetime import datetime
from geopy import distance

# Seleção de modelos
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import train_test_split

# Modelos
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Métricas
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Visualização
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Parâmetros importantes
# Gerando seed para replicação dos resultados
seed = np.random.seed(42)

# Funções criadas para facilitar o processo de análise
# Funções para verificar observações que não estão dentro do intervalo esperado para latitude e longitude
def checkLatitude(df, columns):
    dfInvalidLat = pd.DataFrame()
    for col in columns:
        dfInvalidLat = pd.concat([dfInvalidLat, df.loc[(df[col] < -90) | (df[col] > 90)]])
        print('Foram encontradas', dfInvalidLat.shape[0], 'inválidas na coluna', col)
    return dfInvalidLat

def checkLongitude(df, columns):
    dfInvalidLon = pd.DataFrame()
    for col in columns:
        dfInvalidLon = pd.concat([dfInvalidLon, df.loc[(df[col] < -180) | (df[col] > 180)]])
        print('Foram encontradas', dfInvalidLon.shape[0], 'inválidas na coluna', col)
    return dfInvalidLon

# Função para verificar observações com latitude e longitude igual a zero
def checkLatLonZero(df, columns):
    dfZero = pd.DataFrame()
    for col in columns:
        dfZero = pd.concat([dfZero, df.loc[df[col] == 0]])
        print('Foram encontradas', dfZero.shape[0], 'inválidas na coluna', col)
    return dfZero

# Removendo os valores de acordo com a medida IQR
def removeOutlierIQR(df, columns):
    lenInicial = df.shape[0]
    for col in columns:
        # Definindo quantis
        q1 = df[col].quantile(0.25)
        q2 = df[col].quantile(0.50)
        q3 = df[col].quantile(0.75)
        
        # Calculando IQR
        IQR = q3 - q1
        print('IQR para', col, IQR)

        # Mantendo apenas valores dentro do IQR
        df = df[(df[col] >= q1 - (IQR * 1.5)) & (df[col] <= q3 + (IQR * 1.5))]

        # Verificando tamanho do dataset
        print('Foram removidas', lenInicial - df.shape[0], 'devido a outliers na coluna', col)
    return df

# Calculando distancia entre dois pontos, https://geopy.readthedocs.io/en/latest/#module-geopy.distance
def calculoDistancia(partida_lat, partida_lon, chegada_lat, chegada_lon):
    origem = (partida_lat, partida_lon)
    destino = (chegada_lat, chegada_lon)
    return(distance.distance(origem, destino).km)

def verificaFeriado(year, datetime):
    feriado = holidays.CountryHoliday('US', state='NY', years = year)
    if(datetime in feriado):
        return 1
    else:
        return 0

# Download dos dados com API Kaggle
# ! kaggle competitions download -c new-york-city-taxi-fare-prediction
# ! unzip new-york-city-taxi-fare-prediction.zip -d train

# Leitura do arquivo
df = pd.read_csv('train.csv')
df.head()

# Criando cópia para segurança do dataset
dados = df.copy()

# Verificando tamanho do dataset
print('Shape do dataset:\n', dados.shape, '\n')

# Verificando tipo dos dados
print('Tipo das colunas:\n', dados.dtypes)

# Limpeza e tratamento dos dados
# Verificando se todas terminam com UTC
np.all(dados['pickup_datetime'].str[-3:] == 'UTC')

# Como todas as observações terminam com UTC, o fim da string será desconsiderado na conversão
dados['pickup_datetime_converted'] = dados['pickup_datetime'].apply(lambda x: datetime.strptime(x[:-4], '%Y-%m-%d %H:%M:%S'))
dados.head()

# Removendo coluna original
dados = dados.drop(columns = ['pickup_datetime'])
dados.head()

# Verificando valores únicos de passenger_count
print('Valores únicos para passenger_count', dados['passenger_count'].unique())

# Verificando as linhas com NaN
dados[dados.isna().any(axis = 1)]

# Removendo as observações que contém NaN
linhas = dados.shape[0]
dados = dados.dropna()
print('Total de observações removidas com NaN:', linhas - dados.shape[0])

# Convertendo tipo da coluna "passenger_count" de float para int
dados['passenger_count'] = dados['passenger_count'].astype(int)
print('Tipo da coluna passenger_count:', dados['passenger_count'].dtype)
print('Valores únicos para passenger_count', dados['passenger_count'].unique())

# Gerando boxplots
dados.boxplot(figsize = [20, 10])

# Verificando estatísticas das colunas
dados.describe()

# Verificando Longitude
checkLongitude(dados, ['dropoff_longitude', 'pickup_longitude'])

# Verificando Latitude
checkLatitude(dados, ['pickup_latitude', 'dropoff_latitude'])

# Removendo as observações com latitude e longitude fora dos intervalos
# Verificando tamanho do dataset
print('Shape do dataset:\n', dados.shape, '\n')

dados = dados.drop(labels = checkLongitude(dados, ['dropoff_longitude', 'pickup_longitude']).index, axis = 0)
dados = dados.drop(labels = checkLongitude(dados, ['pickup_latitude', 'dropoff_latitude']).index, axis = 0)

# Verificando tamanho do dataset
print('Shape do dataset:\n', dados.shape, '\n')

# Verificando valores únicos de passageiros
dados['passenger_count'].value_counts()

# Verificando observações com 9 e 208
dados.loc[(dados['passenger_count'] == 9) | (dados['passenger_count'] == 208)]

# Removendo observações com passegeiros = 208
dados = dados.drop(labels = dados.loc[dados['passenger_count'] == 208].index, axis = 0)

# Amount menor ou igual a zero 
dados.loc[dados['fare_amount'] <= 0]

# Removendo os valores negativos e iguais a zero
dados = dados.drop(labels = (dados.loc[dados['fare_amount'] <= 0]).index, axis = 0)

# Verificando tamanho do dataset
print('Shape do dataset:\n', dados.shape, '\n')

# Removendo os outliers segundo o IQR
dados = removeOutlierIQR(dados, ['fare_amount'])

# Verificando observações com latitude e longitude iguais a zero
checkLatLonZero(dados, ['dropoff_longitude', 'pickup_longitude', 'pickup_latitude', 'dropoff_latitude'])

# Removendo observações com latitude e longitude zeradas
dados = dados.drop(labels = checkLatLonZero(dados, ['dropoff_longitude', 'pickup_longitude', 'pickup_latitude', 'dropoff_latitude']).index, axis = 0)

# Verificando tamanho do dataset
print('Shape do dataset:\n', dados.shape, '\n')

# Gerando boxplots após remoção de latitude e longitude zerados
dados.boxplot(figsize = [20, 10])

# Removendo outliers de Latitude e Longitude segundo IQR
dados = removeOutlierIQR(dados, ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'])

# Verificando tamanho do dataset
print('Shape do dataset:\n', dados.shape, '\n')

# Análise Exploratória
# Correlação entre as variáveis
corr = dados.corr()
corr

# Criando nova coluna com base no calculo da distancia
dados['distancia'] = dados.apply(lambda x: calculoDistancia(x['pickup_latitude'], 
                                                            x['pickup_longitude'],
                                                            x['dropoff_latitude'], 
                                                            x['dropoff_longitude']), axis = 1)
dados.head()

# Criando nova coluna com os minutos do dia em que o passageiro entrou no taxi
dados['pickup_horario'] = dados['pickup_datetime_converted'].apply(lambda x: x.hour/60 + x.minute)
dados.head()

# Criando coluna de feriado para todo o estado de NY
dados['feriado'] = dados['pickup_datetime_converted'].apply(lambda x: verificaFeriado(x.year, x))
dados['feriado'].unique()

# Criando coluna de dia da semana ou não
dados['dia_semana'] = dados['pickup_datetime_converted'].apply(lambda x: x.weekday())
dados['dia_semana'].unique()

# Correlação após criação das novas colunas
corr = dados.corr()
corr

# Exibindo correlação com mapa de calor
plt.figure(figsize = (9, 7))
plt.imshow(corr, cmap = 'BuPu', interpolation = 'none', aspect = 'auto')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation = 'vertical')
plt.yticks(range(len(corr)), corr.columns);
plt.suptitle('Correlação entre as variáveis', fontsize = 12)
plt.grid(False)
plt.show()

# Verificando o histograma da variável resposta
sns.histplot(data = dados, x = 'fare_amount', kde = True)

# Verificando quantis teóticos para a variável resposta
stats.probplot(dados['fare_amount'], plot = plt)

# Treinamento dos modelos
# Split dos dados
X = dados[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
           'dropoff_latitude', 'passenger_count', 'distancia',
           'pickup_horario', 'feriado', 'dia_semana']]

y = dados['fare_amount']

p = 0.2
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = p, random_state = seed)

print('Tamanho treino:', x_train.shape)
print('Tamanho teste:', x_test.shape)

# Random Forest Regressor
# Estimando os de melhores parametros com GridSearch
parametros = {
    'max_depth': range(3, 5),
    'n_estimators': (10, 50, 100),
}
gsc = GridSearchCV(estimator = RandomForestRegressor(),
                   param_grid = parametros, 
                   cv = 5, 
                   scoring = 'neg_mean_squared_error')
    
grid_result = gsc.fit(x_train, y_train)
best_params = grid_result.best_params_

print('Melhores parametros para RandomForest', best_params)

# Criando o modelo com os os melhores parâmetros estimados segundo o o GridSearch
model_rf = RandomForestRegressor(max_depth = best_params['max_depth'],
                                 n_estimators = best_params['n_estimators'], 
                                 random_state = seed)

# Verificando tempo de execução do processamento
inicio = time.time()

model_rf.fit(x_train, y_train)

# Prevendo valores de teste
y_pred = model_rf.predict(x_test)

# Avaliação de MSE, RMSE e Coeficiente de Determinação (R2)
EQM = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

fim = time.time()
print('Duração desde da estimação até exibição dos resultados:', fim - inicio)

print('Avaliando o modelo:\n', 
      'EQM: ', EQM,
      'R2: ', r2)

# Analisando os atributos mais importantes para o modelo
feature_names = dados[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
           'dropoff_latitude', 'passenger_count', 'distancia',
           'pickup_horario', 'feriado', 'dia_semana']].columns

print('Importância dos atributos: ', model_rf.feature_importances_)

plt.barh(feature_names, model_rf.feature_importances_)

# Regressão Linear Múltipla
# Utilizando todas as variáveis, inclusive a qualidade
lm = LinearRegression()
lm.fit(x_train, y_train)

y_pred_lm = lm.predict(x_test)

# Avaliando o erro quadrático médio
EQM = mean_squared_error(y_test, y_pred_lm)
print("EQM Regressão Linear Multipla:", EQM)

# Avaliando r2
R2 = r2_score(y_test, y_pred_lm)
print('R2 Regressão Linear Multipla:', R2)

# Observando os resíduos, eles parecem se comprotar conforme o esperado:
# Tes uma distribuição normal com média próxima de zero
res = y_test - y_pred_lm
sns.histplot(data = res,  kde = True)

# Dispersão dos resíduos
sns.scatterplot(y = res, x = y_pred_lm)

plt.hlines(y = 0, xmin = 0, xmax = 30, color = 'red')
plt.ylabel('$\epsilon = y - \hat{y}$ - Resíduos')
plt.xlabel('$\hat{y}$ ou $E(y)$ - Predito')
plt.show()

# Correlação entre o y e y predito
# Espero que não haja nenhuma
corr_y = pd.Series(res).corr(pd.Series(y_pred_lm))
print('Correlação entre resíduos e y predito', corr_y)