#1) Importação de bibliotecas

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import  DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

#-----------------------------------------------------------------------------------------------------------------------------
#2)carregamento das databases

baseTreinamento = pd.read_csv("train.csv")
baseTeste = pd.read_csv("test.csv")

#3)Tratamento das variáveis categóricas

#Colunas que contém variáveis categóricas
colunasParaTransformacaoPorIndice= [2,5,7,8,9,10,11,12,13,14,15,16,21,22,23,24,25,27,28,29,30,31,32,33,35,
                            39,40,41,42,53,55,57,58,60,63,64,65,72,73,74,78,79] 
colunasParaTransformacaoPorNome= ['MSZoning',
                            'Street',
                            'Alley',
                            'LotShape',
                            'LandContour',
                            'Utilities',
                            'LandSlope',
                            'Neighborhood',
                            'Condition1',
                            'Condition2',
                            'BldgType',
                            'HouseStyle',
                            'RoofStyle',
                            'RoofMatl',
                            'Exterior1st',
                            'Exterior2nd',
                            'MasVnrType',
                            'ExterQual',
                            'ExterCond',
                            'Foundation',
                            'BsmtQual',
                            'BsmtCond',
                            'BsmtExposure',
                            'BsmtFinType1',
                            'BsmtFinType2',
                            'PoolQC',
                            'Fence',
                            'MiscFeature',
                            'SaleType',
                            'SaleCondition',
                            'LotConfig',
                            'Heating',
                            'HeatingQC',
                            'CentralAir',
                            'Electrical',
                            'KitchenQual',
                            'Functio0l',
                            'FireplaceQu',
                            'GarageType',
                            'GarageFinish',
                            'GarageQual',
                            'GarageCond',
                            'PavedDrive'
                            ]

#Realização da transformação das Variáveis categóricas pelo LabelEncoder
labelencoder = LabelEncoder()
baseTreinamento['MSZoning'] = labelencoder.fit_transform(baseTreinamento['MSZoning'])
baseTreinamento['Alley'] = labelencoder.fit_transform(baseTreinamento['Alley'])
baseTreinamento['Street'] = labelencoder.fit_transform(baseTreinamento['Street'])
baseTreinamento['LotShape'] = labelencoder.fit_transform(baseTreinamento['LotShape'])
baseTreinamento['LandContour'] = labelencoder.fit_transform(baseTreinamento['LandContour'])
baseTreinamento['Utilities'] = labelencoder.fit_transform(baseTreinamento['Utilities'])
baseTreinamento['LandSlope'] = labelencoder.fit_transform(baseTreinamento['LandSlope'])
baseTreinamento['Neighborhood'] = labelencoder.fit_transform(baseTreinamento['Neighborhood'])
baseTreinamento['Condition1'] = labelencoder.fit_transform(baseTreinamento['Condition1'])
baseTreinamento['Condition2'] = labelencoder.fit_transform(baseTreinamento['Condition2'])
baseTreinamento['BldgType'] = labelencoder.fit_transform(baseTreinamento['BldgType'])
baseTreinamento['HouseStyle'] = labelencoder.fit_transform(baseTreinamento['HouseStyle'])
baseTreinamento['RoofStyle'] = labelencoder.fit_transform(baseTreinamento['RoofStyle'])
baseTreinamento['RoofMatl'] = labelencoder.fit_transform(baseTreinamento['RoofMatl'])
baseTreinamento['Exterior1st'] = labelencoder.fit_transform(baseTreinamento['Exterior1st'])
baseTreinamento['Exterior2nd'] = labelencoder.fit_transform(baseTreinamento['Exterior2nd'])
baseTreinamento['MasVnrType'] = labelencoder.fit_transform(baseTreinamento['MasVnrType'])
baseTreinamento['ExterQual'] = labelencoder.fit_transform(baseTreinamento['ExterQual'])
baseTreinamento['ExterCond'] = labelencoder.fit_transform(baseTreinamento['ExterCond'])
baseTreinamento['Foundation'] = labelencoder.fit_transform(baseTreinamento['Foundation'])
baseTreinamento['BsmtQual'] = labelencoder.fit_transform(baseTreinamento['BsmtQual'])
baseTreinamento['BsmtCond'] = labelencoder.fit_transform(baseTreinamento['BsmtCond'])
baseTreinamento['BsmtExposure'] = labelencoder.fit_transform(baseTreinamento['BsmtExposure'])
baseTreinamento['BsmtFinType1'] = labelencoder.fit_transform(baseTreinamento['BsmtFinType1'])
baseTreinamento['BsmtFinType2'] = labelencoder.fit_transform(baseTreinamento['BsmtFinType2'])
baseTreinamento['PoolQC'] = labelencoder.fit_transform(baseTreinamento['PoolQC'])
baseTreinamento['Fence'] = labelencoder.fit_transform(baseTreinamento['Fence'])
baseTreinamento['MiscFeature'] = labelencoder.fit_transform(baseTreinamento['MiscFeature'])
baseTreinamento['SaleType'] = labelencoder.fit_transform(baseTreinamento['SaleType'])
baseTreinamento['SaleCondition'] = labelencoder.fit_transform(baseTreinamento['SaleCondition'])
baseTreinamento['LotConfig'] = labelencoder.fit_transform(baseTreinamento['LotConfig'])
baseTreinamento['Heating'] = labelencoder.fit_transform(baseTreinamento['Heating'])
baseTreinamento['HeatingQC'] = labelencoder.fit_transform(baseTreinamento['HeatingQC'])
baseTreinamento['CentralAir'] = labelencoder.fit_transform(baseTreinamento['CentralAir'])
baseTreinamento['Electrical'] = labelencoder.fit_transform(baseTreinamento['Electrical'])
baseTreinamento['KitchenQual'] = labelencoder.fit_transform(baseTreinamento['KitchenQual'])
baseTreinamento['Functio0l'] = labelencoder.fit_transform(baseTreinamento['Functio0l'])
baseTreinamento['FireplaceQu'] = labelencoder.fit_transform(baseTreinamento['FireplaceQu'])
baseTreinamento['GarageType'] = labelencoder.fit_transform(baseTreinamento['GarageType'])
baseTreinamento['GarageFinish'] = labelencoder.fit_transform(baseTreinamento['GarageFinish'])
baseTreinamento['GarageQual'] = labelencoder.fit_transform(baseTreinamento['GarageQual'])
baseTreinamento['GarageCond'] = labelencoder.fit_transform(baseTreinamento['GarageCond'])
baseTreinamento['PavedDrive'] = labelencoder.fit_transform(baseTreinamento['PavedDrive'])


# # # #Definindo as colunas que farão o treinamento e a coluna de respostas da baseTreinamento
#Previsores
X = baseTreinamento.iloc[:,0:80]
#Classe
Y = baseTreinamento.iloc[:,80:81]

# #Transformação das Variaveis categóricas com o one hot encoder da variavel x que contem os previsores
# onehot = OneHotEncoder(handle_unknown='ignore')
# onehot.fit_transform(X['MSZoning'].reshape(-1,1))

# #Divisão em da basetreinamento em treinamento e teste para criar nosso modelo regressor
Xtreinamento,Xteste,YTreinamento,Yteste = train_test_split(X,Y,
                                                            test_size = 0.3,
                                                            random_state = 0)



#Tratamento das Variaveis categoricas da baseTeste
for i in colunasParaTransformacaoPorNome:
    baseTeste[i] = labelencoder.fit_transform(baseTeste[i])
    
#------------------------------------------------------------------------------------------------------------------------------
#3)Aplicação dos métodos de Machine Learning
#------------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------------
#4 - Regressão Multipla simples com Scikit Learn 
#------------------------------------------------------------------------------------------------------------------------------

# # # # #Realizando a criação do modelo de regressao multipla 
# regressor = LinearRegression()
# regressor.fit(Xtreinamento,YTreinamento)
# score = regressor.score(Xtreinamento,YTreinamento)

# # # # #realização das previsões com o modelo criado n a baseTreinamento 
# previsoesDotreinamento = regressor.predict(Xteste)

# # # # #Realizando a avaliação das previsões feitas pelo modelo e verificando seu erro 
# meanAbsoluteError = mean_absolute_error(Yteste,previsoesDotreinamento)
# print(meanAbsoluteError)
# regressor.score(Xteste,Yteste)

# # # # #Realizando as previsões dos valores das casas com o modelo criado, por meio da baseTeste
# # # previsõesDoteste = regressor.predict(baseTeste)

# # # # # #Avaliação Final do algoritmo
# # # # # #Precisão de 0.6440896204748197 de acertos
# # # # # #Erro absoluto de 22473.583392859135 dolares pra mais ou pra menos  


# # #------------------------------------------------------------------------------------------------------------------------------
# # #5)Regressão Por arvore de Decisão 
# # #------------------------------------------------------------------------------------------------------------------------------


# # #Criação do Modelo de regressão por decision tree

# regressorTree = DecisionTreeRegressor()
# regressorTree.fit(Xtreinamento,YTreinamento)
# scoreTree = regressorTree.score(Xtreinamento,YTreinamento)
# previsoesDotreinamentoDecisionTree = regressorTree.predict(Xteste)

# # #Calculo do Erro absoluro do Modelo Decision Tree
# meanAbsoluteErrorDecisionTree = mean_absolute_error(Yteste,previsoesDotreinamentoDecisionTree)

# # # # # # #Avaliação Final do algoritmo
# # # # # # #Precisão de 0.7702424572699048 de acertos
# # # # # # #Erro absoluto de 26804.812785388127dolares pra mais ou pra menos  
# # # # # # Temos um erro ligeiramente maior mas com a mesma precisão da multipla 

# # #------------------------------------------------------------------------------------------------------------------------------
# # #5)Regressão Por Ramdom Forest
# # #------------------------------------------------------------------------------------------------------------------------------

# regressorRandom = RandomForestRegressor(n_estimators = 10)
# regressorRandom.fit(Xtreinamento,YTreinamento)
# scoreRf = regressorRandom.score(Xtreinamento,YTreinamento)

# previsoesDotreinamentoRandomForest = regressorRandom.predict(Xteste)

# meanAbsoluteErrorRF = mean_absolute_error(Yteste,previsoesDotreinamentoRandomForest)

# # # # # # #Avaliação Final do algoritmo
# # # # # # #Precisão de 0.9704050776456907 de acertos
# # # # # # #Erro absoluto de 18970.488356164384 dolares pra mais ou pra menos  
# # # # # # Temos um erro ligeiramente maior mas com a mesma precisão da multipla 


# # #------------------------------------------------------------------------------------------------------------------------------
# # #5)Regressão Por Ramdom Forest
# # #------------------------------------------------------------------------------------------------------------------------------

regressorSVR = SVR(kernel = 'rbf')
regressorSVR.fit(Xtreinamento,YTreinamento)
scoreSVR = regressorSVR.score(Xtreinamento,YTreinamento)
previsoesDotreinamentoSVR= regressorSVR.predict(Xteste)
meanAbsoluteErrorSVR = mean_absolute_error(Yteste,previsoesDotreinamentoSVR)

