#--------------------------------------------------------------------------------------------------------------
# Importação dos dados 
#--------------------------------------------------------------------------------------------------------------
import pandas as pd

# Importando a database do Kaggle
database = pd.read_csv('creditcard.csv')


#--------------------------------------------------------------------------------------------------------------
# Pré processamento da database 
#--------------------------------------------------------------------------------------------------------------

# Como os dados estão muito desbalanceados temos localizacao das fraudes e das nao fraudes
fraudes=database.loc[database['Class']==1]
#Temos 492 Fraudes no total 
naoFraude = database.loc[database['Class']==0]
#Como existem 284315 não fraudes vamos escolher 500 dados nao frudes para trabalhar
#naoFraude= naoFraude[1000:3000]


#Criando a database balanceada
databaseB = pd.concat([fraudes,naoFraude])

# Determinando quais colunas são os nossos previsores e qual a coluna de Classe
previsores = databaseB.drop('Class',1)
classe = databaseB['Class']

# Escalonando os valores dos previsores

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Divisão da database em treimento e teste 

from sklearn.model_selection import train_test_split
previsoresTreinamento, previsoresTeste, classeTreinamento, classeTeste = train_test_split(previsores,classe,test_size=0.3,random_state = 0)

#--------------------------------------------------------------------------------------------------------------
# Realizando a classificação pelo algoritmo de Naive Bayes  OK
#--------------------------------------------------------------------------------------------------------------

# from sklearn.naive_bayes import GaussianNB

# # Realizando o treinamento do algoritmo

# classificadorBayes = GaussianNB()
# classificadorBayes.fit(previsoresTreinamento,classeTreinamento)

# # Realizando o teste do algoritmo

# resultadosBayes = classificadorBayes.predict(previsoresTeste)

# # Comparando os resultados previstos com a classeTeste

# from sklearn.metrics import confusion_matrix,accuracy_score
# precisaoBayes = accuracy_score(classeTeste,resultadosBayes)
# matrizBayes = confusion_matrix(classeTeste,resultadosBayes)

# #Precisão de 0.9786641386655431 com 1823 erros

#--------------------------------------------------------------------------------------------------------------
# Realizando a classificação pelo algoritmo de Arvore de Decisão  OK
#--------------------------------------------------------------------------------------------------------------

# from sklearn.tree import DecisionTreeClassifier
# classificadorTree = DecisionTreeClassifier(criterion='entropy', random_state=0)
# classificadorTree.fit(previsoresTreinamento, classeTreinamento)
# previsoesTree = classificadorTree.predict(previsoresTeste)

# from sklearn.metrics import confusion_matrix, accuracy_score
# precisaoTree = accuracy_score(classeTeste, previsoesTree)
# matrizTree = confusion_matrix(classeTeste, previsoesTree)

# from sklearn.tree import export
# export.export_graphviz(classificadorTree,
#                        out_file ='arvore.dot', 
#                        feature_names = ['Time',	'V1',	'V2',	'V3',	'V4',	'V5',	'V6',	'V7',	'V8',	'V9',	'V10',	'V11',	'V12',	'V13',	'V14',	'V15',	'V16',	'V17',	'V18',	'V19',	'V20',	'V21',	'V22',	'V23',	'V24',	'V25',	'V26',	'V27',	'V28',	'Amount'],
#                        class_names = ['Fraude','NaoFraude'],
#                        filled = True,
#                        leaves_parallel = True)

# #Precisão de 0.9991924440855307 com 69 erros 

#--------------------------------------------------------------------------------------------------------------
# Realizando a classificação pelo algoritmo de Random Forest
#--------------------------------------------------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
classificadorRF = RandomForestClassifier(n_estimators=10,
                                         criterion='entropy',
                                         random_state=0)
classificadorRF.fit(previsoresTreinamento,classeTreinamento)
previsoesRF = classificadorRF.predict(previsoresTeste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisaoRF = accuracy_score(classeTeste, previsoesRF)
matrizRF = confusion_matrix(classeTeste, previsoesRF)

#Precisão de 0.9994499256814484 com 44 erros 