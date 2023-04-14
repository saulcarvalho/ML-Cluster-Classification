class npArr:
    """ 
    DESC: Função que recebe um NumPy Array 'df', verifica as colunas indicadas pelo range 'rg' e 
          substitui os valores NaN pela média da respetiva coluna. Devolve o array já corrigido.
          -------------------------------------------------------------------
          (IN):   df      - NumPy Array
                  rg      - range() das colunas de df a analisar
          (OUT):  out     - Fixed NumPy Array
    """
    def NaN2Mean(df,rg):
        for i in rg:
            counter = 0 
            temp = df[:,i];
        
            for idx, x in enumerate(temp):    # Encontra os campos com valor NaN na coluna e apaga-os para conseguir depois calcular a média.
                if np.isnan(x):
                    temp = np.delete(temp,idx - counter)
                    counter = counter + 1
        
            mean = np.mean(temp)
            
            d = df[:,i]
            for idx, x in enumerate(d):       # Substitui os campos com valor NaN pela média da coluna
                if np.isnan(x):
                    d[idx] = mean
            
            df[:,i] = d
            
        out = df
        
        return out
    
    """ 
    DESC: Função que recebe um NumPy Array 'df', verifica as colunas indicadas pelo range 'rg' e 
          aplica normalização standard ao array. Devolve o NumPy Array já normalizado.
          -------------------------------------------------------------------
          (IN):   df      - NumPy Array
                  rg      - range() das colunas de df a normalizar
          (OUT):  out     - Std. Normalized NumPy Array
    """
    def std_normalization(df,rg):
        for i in rg:
            temp = df[:,i];
            mean = np.mean(temp)
            std  = np.std(temp)     
            
            d = (temp - mean)/std    # Aplica normalização standard
            df[:,i] = d
            
        out = df
        
        return out
    
    """ 
    DESC: Função que recebe um NumPy Array 'df', verifica as colunas indicadas pelo range 'rg' e 
          elimina a respetiva linha de todo o dataset  da respetiva coluna. Devolve o array já corrigido.
          -------------------------------------------------------------------
          (IN):   df      - NumPy Array
                  rg      - range() das colunas de df a analisar para outliers
          (OUT):  out     - NumPy Array sem outliers
    """
    def outlier_remover(df,rg):
        temp = df;
        
        for i in rg:
            z = stats.zscore(dataset[:,i].astype(float))    
            outlier = np.where(z > 3)                       # Vetor com indices dos outliers

            for idx, x in enumerate(outlier):    
                temp = np.delete(temp, x, 0)                

        out = temp
        
        return out

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# INÍCIO DO CÓDIGO
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

KNN_sel = True
ANN_sel = False
SVM_sel = False


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import pyautogui
from scipy import stats
from sklearn import preprocessing
import seaborn as sns


# Apaga os plots guardados no Spyder. «!» Só funciona se a respetiva janela 'Plots' 
# estiver aberta e selecionada no canto superior direito
pyautogui.moveTo(1500, 350)                 # Movimenta cursor para X e Y
pyautogui.click()                           # Mouse click
with pyautogui.hold('ctrl'):                # pressiona Ctrl+Shift+W
    with pyautogui.hold('shit'):    
        pyautogui.press('W')

dataset = pd.read_excel('Datasets.xlsx', sheet_name='Classification')
ds_size = dataset.size
dataset = np.array(dataset)
dataset = dataset[:,0:10]



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# PREPARAÇÃO DO DATASET
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

#---------------------------------------------
# CORREÇÃO DO DATASET
dataset = npArr.NaN2Mean(dataset, range(2,3)) # Correção de valores NaN para média da respetiva coluna
dataset = npArr.NaN2Mean(dataset, range(7,9)) # Correção de valores NaN para média da respetiva coluna

#---------------------------------------------
# PLOTS 
labels = range(1, ds_size) 
plt.rcParams['lines.linewidth'] = 1

# Plot da feature Dependents
fig1 = plt.figure(figsize=(10,10))
ax = fig1.add_subplot(111)      
plt.title("Feature Dependents")                
plt.xlabel("ID")
plt.ylabel("Dependents")
ax.scatter(dataset[:,0], dataset[:,1], label='True Position')
plt.grid(True) 
plt.show() 

# Plot da feature Tenure
fig1 = plt.figure(figsize=(10,10))
ax = fig1.add_subplot(211)      
plt.title("Feature Tenure")                
plt.xlabel("ID")
plt.ylabel("Tenure")
ax.scatter(dataset[:,0], dataset[:,2], label='True Position')
plt.grid(True) 
for label, x, y in zip(labels, dataset[:,0], dataset[:,2]):   
    plt.annotate(label, xy=(x,y), xytext=(-3,3), textcoords='offset points', ha='right', va='bottom')

ax = fig1.add_subplot(212)   
plt.title("Feature Tenure")                
plt.xlabel("Tenure")
ax.boxplot(dataset[:,2], vert=False)
plt.grid(False)
plt.show() 
    
# Plot da feature InternetService
fig1 = plt.figure(figsize=(10,10))
ax = fig1.add_subplot(111)      
plt.title("Feature InternetService")                
plt.xlabel("ID")
plt.ylabel("InternetService")
ax.scatter(dataset[:,0], dataset[:,3], label='True Position')
plt.grid(True) 
plt.show() 

# Plot da feature StreamingMovies
fig1 = plt.figure(figsize=(10,10))
ax = fig1.add_subplot(111)      
plt.title("Feature StreamingMovies")                
plt.xlabel("ID")
plt.ylabel("StreamingMovies")
ax.scatter(dataset[:,0], dataset[:,4], label='True Position')
plt.grid(True) 
plt.show() 

# Plot da feature Contract
fig1 = plt.figure(figsize=(10,10))
ax = fig1.add_subplot(111)      
plt.title("Feature Contract")                
plt.xlabel("ID")
plt.ylabel("Contract")
ax.scatter(dataset[:,0], dataset[:,5], label='True Position')
plt.grid(True) 

# Plot da feature PaymentMethod
fig1 = plt.figure(figsize=(10,10))
ax = fig1.add_subplot(211)      
plt.title("Feature PaymentMethod")                
plt.xlabel("ID")
plt.ylabel("PaymentMethod")
ax.scatter(dataset[:,0], dataset[:,6], label='True Position')
plt.grid(True) 
plt.show() 

# Plot da feature MonthlyCharges
fig2 = plt.figure(figsize=(10,10))
ax = fig2.add_subplot(211)         
plt.title("Feature MonthlyCharges")          
plt.xlabel("ID")
plt.ylabel("MonthlyCharges")
ax.scatter(dataset[:,0], dataset[:,7], label='True Position') 
plt.grid(True) 
for label, x, y in zip(labels, dataset[:,0], dataset[:,7]):
    plt.annotate(label, xy=(x,y), xytext=(-3,3), textcoords='offset points', ha='right', va='bottom')

ax = fig2.add_subplot(212)   
plt.title("Feature MonthlyCharges")                     
plt.xlabel("MonthlyCharges")
ax.boxplot(dataset[:,7], vert=False)
plt.grid(False)
plt.show() 

# Plot da feature TotalCharges
fig3 = plt.figure(figsize=(10,10))
ax = fig3.add_subplot(211)          
plt.title("Feature TotalCharges")               
plt.xlabel("ID")
plt.ylabel("TotalCharges")
ax.scatter(dataset[:,0], dataset[:,8], label='True Position')
plt.grid(True) 
for label, x, y in zip(labels, dataset[:,0], dataset[:,8]): 
    plt.annotate(label, xy=(x,y), xytext=(-3,3), textcoords='offset points', ha='right', va='bottom')
      
ax = fig3.add_subplot(212)       
plt.title("Feature TotalCharges")             
plt.xlabel("TotalCharges")
ax.boxplot(dataset[:,8], vert=False)
plt.grid(False)
plt.show()


# Plot do target Churn
fig3 = plt.figure(figsize=(10,10))
ax = fig3.add_subplot(211)          
plt.title("Target Churn")               
plt.xlabel("ID")
plt.ylabel("Churn")
ax.scatter(dataset[:,0], dataset[:,9], label='True Position')
plt.grid(True) 
plt.show() 
print("\n\n\n\n\n")

#---------------------------------------------
# REMOÇÃO OUTLIERS
dataset = npArr.outlier_remover(dataset, range(7,9))

# Plot da feature MonthlyCharges (sem outliers)
fig2 = plt.figure(figsize=(10,10))
ax = fig2.add_subplot(211)         
plt.title("Feature MonthlyCharges")          
plt.xlabel("ID")
plt.ylabel("MonthlyCharges")
ax.scatter(dataset[:,0], dataset[:,7], label='True Position') 
plt.grid(True) 
for label, x, y in zip(labels, dataset[:,0], dataset[:,7]):
    plt.annotate(label, xy=(x,y), xytext=(-3,3), textcoords='offset points', ha='right', va='bottom')

ax = fig2.add_subplot(212)   
plt.title("Feature MonthlyCharges")                     
plt.xlabel("MonthlyCharges")
ax.boxplot(dataset[:,7], vert=False)
plt.grid(False)
plt.show() 

# Plot da feature TotalCharges (sem outliers)
fig3 = plt.figure(figsize=(10,10))
ax = fig3.add_subplot(211)          
plt.title("Feature TotalCharges")               
plt.xlabel("ID")
plt.ylabel("TotalCharges")
ax.scatter(dataset[:,0], dataset[:,8], label='True Position')
plt.grid(True) 
for label, x, y in zip(labels, dataset[:,0], dataset[:,8]): 
    plt.annotate(label, xy=(x,y), xytext=(-3,3), textcoords='offset points', ha='right', va='bottom')
    
ax = fig3.add_subplot(212)       
plt.title("Feature TotalCharges")             
plt.xlabel("TotalCharges")
ax.boxplot(dataset[:,8], vert=False)
plt.grid(False)
plt.show()

#---------------------------------------------
# NORMALIZAÇÃO
dataset_n = npArr.std_normalization(dataset, range(2,3))
dataset_n = npArr.std_normalization(dataset, range(7,9))


# Extração das features + target
data = dataset_n[:,1:10]
xlim = []
xlim.append(np.min(dataset_n[:,1]))
xlim.append(np.max(dataset_n[:,1]))
ylim = []
ylim.append(np.min(dataset_n[:,6]))
ylim.append(np.max(dataset_n[:,6]))

# Codificação das features e targets com campos string 
enc = preprocessing.LabelEncoder()
Dependents_enc      = enc.fit_transform(data[:,0])
InternetService_enc = enc.fit_transform(data[:,2]) 
StreamingMovies_enc = enc.fit_transform(data[:,3]) 
Contract_enc        = enc.fit_transform(data[:,4]) 
PaymentMethod_enc   = enc.fit_transform(data[:,5])

Churn_enc = enc.fit_transform(data[:,8])

# Redifinição do dataset
data_enc = np.vstack((Dependents_enc, data[:,1], InternetService_enc, StreamingMovies_enc, Contract_enc, PaymentMethod_enc,
                      data[:,6], data[:,7], Churn_enc)).T
data_enc = data_enc.astype(float)

# Seleção de features por filtragem
df = pd.DataFrame(data_enc, columns = ['Dependents', 'Tenure', 'InternetService', 'StreamingMovies', 
                                   'Contract', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'])
features_data = df.drop('Churn',1)
target_data   = df['Churn']
plt.figure(figsize=(16,14))
corr = df.corr(method='pearson')
sns.heatmap(corr, annot=True, cmap=plt.cm.Reds)
plt.show()

corr_target = abs(corr['Churn'])                            # Seleciona features irrelevantes (< 0.15)
irrelevant_features = corr_target[corr_target < 0.15]

for idx, x in enumerate(irrelevant_features):               # Remoção das features irrelevantes do dataset
    df = df.drop(irrelevant_features.index[idx],1)

data_enc = pd.DataFrame.to_numpy(df)

# Total dados subdividido
features_data   = data_enc[:, 0:4].astype('int')                           
target_data     = data_enc[:, 4].astype('int')

# trainvaling/Validation=64%/16% e Test=20%
f1 = round(0.80 * data_enc[:,0].size)
f2 = round(0.20 * data_enc[:,0].size)
f3 = round(0.16 * f1)

# Preparar os subconjuntos para teste
trainval_data   = data_enc[0 : f1, 0:5].astype('int')
test_data       = data_enc[f1 : f1+f2, 0:5].astype('int')
# Separa validação do train
train_data      = data_enc[0 : f1-f3, 0:5].astype('int')
validation_data = data_enc[f1-f3 : f1, 0:5].astype('int')

# Preparar labels (target) de cada subconjunto de teste
trainval_labels   = data_enc[0 : f1, 4].astype('int')
test_labels       = data_enc[f1 : f1+f2, 4].astype('int')
# Separa validação do train
train_labels      = data_enc[0 : f1-f3, 4].astype('int')
validation_labels = data_enc[f1-f3 : f1, 4].astype('int')


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn import metrics
from colored import fg
import warnings
import math
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# k-NEAREST NEIGHBORS
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
if KNN_sel:
    print("***********************************************************")
    print("*                   k-NEAREST NEIGHBORS                   *")
    print("***********************************************************")

    warnings.filterwarnings('ignore')
    
    n_neighbors   = range(1,21)
    dist_sel      = ['chebyshev', 'euclidean', 'manhattan', 'minkowski']
    weight_sel    = ['distance', 'uniform']
    
    for m in dist_sel:
        for w in weight_sel:
            print(fg(13) + "\n##### MÉTRICA DISTÂNCIA=" + m.upper() + " #####" + fg('white'))
            train_scores = []
            val_scores = []
            grid_train_scores = []
            grid_test_scores = []
            cv_res = []
            for n in n_neighbors:
                print(fg(13) + "\n##### WEIGHTS=" + w.upper() + ", K=" + str(n) + " #####" + fg('white'))
                
                # Treino do modelo
                knn = KNeighborsClassifier(n_neighbors=n, metric=m, weights=w) 
                knn.fit(trainval_data, trainval_labels)
        
                # Predição dos dados de treino
                pred_trainval_label = knn.predict(trainval_data)
                print(fg(11) + "--- DADOS DE TREINO (80%) ---" + fg('white'))
                cm = confusion_matrix(trainval_labels, pred_trainval_label)
                fig = plt.figure(figsize=(25,30))       
                ax = fig.add_subplot(421)
                ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
         
                plt.xlabel('Predições', fontsize=18)
                plt.ylabel('Atuais', fontsize=18)                  
                plt.title("kNN Matriz de confusão Treino (k=" + str(n) + ", dist=" + m + ", weights=" + w + ")")
    
                print(fg('light_gray') + "SCORE DE PRECISÃO: " + fg('white'), end='')
                print (accuracy_score(trainval_labels, pred_trainval_label))
                print(fg('light_gray') + "RELATÓRIO DE CLASSIFICAÇÃO:" + fg('white'))
                print (classification_report(trainval_labels, pred_trainval_label), end='')
        
                # Predição dos dados de teste
                pred_test_label = knn.predict(test_data)
                print(fg(11) + "\n--- DADOS DE TESTE (20%) ---" + fg('white'))
                cm = confusion_matrix(test_labels, pred_test_label)   
                ax = fig.add_subplot(423)
                ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
         
                plt.xlabel('Predições', fontsize=18)
                plt.ylabel('Atuais', fontsize=18)                  
                plt.title("kNN Matriz de confusão Teste (k=" + str(n) + ", dist=" + m + ", weights=" + w + ")")
                
                print(fg('light_gray') + "SCORE DE PRECISÃO: " + fg('white'), end='')
                print (accuracy_score(test_labels, pred_test_label))
                print(fg('light_gray') + "RELATÓRIO DE CLASSIFICAÇÃO:" + fg('white'))
                print (classification_report(test_labels, pred_test_label), end='')
                
                # Plot real vs predicted
                x_ax = range(0, trainval_labels.size)
                ax = fig.add_subplot(422)
                plt.scatter(x_ax, trainval_labels, s=15, color="blue", label="original treino")
                plt.plot(x_ax, pred_trainval_label, ls='--', lw=1.5, color="red", label="predição treino")
                plt.xlabel('Amostras', fontsize=18)                
                plt.title("kNN Original vs previsão Treino (k=" + str(n) + ", dist=" + m + ", weights=" + w + ")")
                plt.legend()
                
                x_ax = range(0, test_labels.size)
                ax = fig.add_subplot(424)
                plt.scatter(x_ax, test_labels, s=15, color="blue", label="original teste")
                plt.plot(x_ax, pred_test_label, ls='--', lw=1.5, color="red", label="predição teste")
                plt.xlabel('Amostras', fontsize=18)                
                plt.title("kNN Original vs previsão Teste (k=" + str(n) + ", dist=" + m + ", weights=" + w + ")")
                plt.legend()
                plt.show()
                
                # Cálculo dos erros
                print(fg(11) + "\n--- MÉTRICAS DE ERRO ---" + fg('white'))
                mse = metrics.mean_squared_error(trainval_labels, pred_trainval_label)
                print(fg('light_gray') + "MSE TREINO: " + fg('white') + str(mse))
                rmse = math.sqrt(mse)
                print(fg('light_gray') + "RMSE TREINO: " + fg('white') + str(rmse))
                mse = metrics.mean_squared_error(test_labels, pred_test_label)
                print(fg('light_gray') + "MSE TESTE: " + fg('white') + str(mse))
                rmse = math.sqrt(mse)
                print(fg('light_gray') + "RMSE TESTE: " + fg('white') + str(rmse))
                
                # Avaliação (3-Fold Split)
                print(fg(11) + "\n--- AVALIAÇÃO 3-FOLD SPLIT ---" + fg('white'))
                knn = KNeighborsClassifier(n_neighbors=n, metric=m, weights=w) 
                knn.fit(train_data, train_labels)
                
                train_scores.append(knn.score(train_data, train_labels))
                val_scores.append(knn.score(validation_data, validation_labels))
                print(fg('light_gray') + "SCORE DE VALIDAÇÃO: " + fg('white') + str(knn.score(validation_data, validation_labels)))
                
                # Avaliação K-Fold CV
                print(fg(11) + "\n--- AVALIAÇÃO K-FOLD CROSS VALIDATION ---" + fg('white'))
                res = cross_validate(knn, trainval_data, trainval_labels, return_train_score=True, cv=5)
                cv_res.append(res)
                print(fg('light_gray') + "SCORE VALIDAÇÃO CRUZADA (CV=5): " + fg('white') + str(np.mean(res['test_score'])))
                
                # Avaliação Grid Search CV
                print(fg(11) + "\n--- AVALIAÇÃO GRID SEARCH CV ---" + fg('white'))
                param_grid = {'n_neighbors':  n_neighbors}
                grid = GridSearchCV(knn, param_grid=param_grid, cv=10, return_train_score=True)
                grid.fit(trainval_data, trainval_labels)
                
                grid_train_scores.append(grid.score(trainval_data, trainval_labels))
                grid_test_scores.append(grid.score(test_data, test_labels))
                print(fg('light_gray') + "SCORE TESTE GRID SEARCH CV (CV=10): " + fg('white') + str(grid.score(test_data, test_labels)))
                
                if n == 10:
                    input(fg('red') + "\n---> Pressionar tecla aleatória para continuar os próximos 10 vizinhos <---" + fg('white'))
                
            print(fg(13) + "\n##### MÉTRICA DISTÂNCIA=" + m.upper() +  ", WEIGHTS=" + w.upper() + " #####" + fg('white'), end='')
            
            # 3-Fold Split
            print(fg(11) + "\n--- RESULTADOS 3-FOLD SPLIT - TREINO/VALIDAÇÃO/TEST (64%/16%/20%) ---" + fg('white'))
            print(fg('light_gray') + "-> MELHOR SCORE DE VALIDAÇÃO: " + fg('white') + str(np.max(val_scores)))
            best_n_neighbors = n_neighbors[np.argmax(val_scores)]
            print(fg('light_gray') + "-> MELHOR NÚMERO DE VIZINHOS: " + fg('white') + str(best_n_neighbors))
            knn = KNeighborsClassifier(n_neighbors=best_n_neighbors, metric=m, weights=w)
            knn.fit(trainval_data, trainval_labels)
            print(fg('light_gray') + "-> MELHOR SCORE DE TESTE:     " + fg('white') + str(knn.score(test_data, test_labels)))
            
            fig = plt.figure(figsize=(15,15))         
            ax = fig.add_subplot(311)
            ax.plot(n_neighbors, train_scores, color='k', marker='o', markerfacecolor='royalblue', markersize=6, label='Treino')
            ax.plot(n_neighbors, val_scores, color='k', marker='o', markerfacecolor='lightskyblue', markersize=6, label='Validação')
            plt.title("kNN 3-Fold Split - Precisão vs k [Treino|Validação] (dist=" + m + ", weights=" + w + ")")
            plt.legend()
            plt.xlabel('k')
            plt.ylabel('Precisão')
            
             
            # K-Fold CV
            print(fg(11) + "\n--- RESULTADOS K-FOLD CROSS VALIDATION (CV) ---" + fg('white'))
            cv_mean = []
            for idx, i in enumerate(cv_res):
                cv_mean.append(np.mean(cv_res[idx]['test_score']))
                
            print(fg('light_gray') + "-> MELHOR SCORE VALIDAÇÃO CRUZADA (CV=5): " + fg('white') + str(np.max(cv_mean)))
            for idx, i in enumerate(cv_mean):
                if cv_mean[idx] == np.max(cv_mean):
                    best_n_neighbors = idx + 1
                    break
            print(fg('light_gray') + "-> MELHOR NÚMERO DE VIZINHOS: " + fg('white') + str(best_n_neighbors))
                 
            ax = fig.add_subplot(312)
            ax.plot(n_neighbors, train_scores, color='k', marker='o', markerfacecolor='royalblue', markersize=6, label='Treino')
            ax.plot(n_neighbors, cv_mean, color='k', marker='o', markerfacecolor='lightskyblue', markersize=6, label='Validação CV')
            plt.title("kNN K-Fold CV (CV=5, dist=" + m + ", weights=" + w + ")")
            plt.legend()
            plt.xlabel('k')
            plt.ylabel('Precisão')
            
            
            # Grid Search CV
            param_grid = {'n_neighbors':  n_neighbors}
            knn = KNeighborsClassifier(metric=m, weights=w) 
            grid = GridSearchCV(knn, param_grid=param_grid, cv=10, return_train_score=True)
            grid.fit(trainval_data, trainval_labels)
            
            print(fg(11) + "\n--- RESULTADOS GRID SEARCH CV ---" + fg('white'))
            print(fg('light_gray') + "-> MELHOR SCORE DE VALIDAÇÃO CRUZADA (CV=10): " + fg('white') + str(grid.best_score_))
            print(fg('light_gray') + "-> MELHOR NÚMERO DE VIZINHOS: " + fg('white') + str(grid.best_params_['n_neighbors']))
            print(fg('light_gray') + "-> MELHOR SCORE DE TESTE:     " + fg('white') + str(grid.score(test_data, test_labels)))
            
            ax = fig.add_subplot(313)
            ax.plot(n_neighbors, grid_train_scores, color='k', marker='o', markerfacecolor='royalblue', markersize=6, label='Treino')
            ax.plot(n_neighbors, grid_test_scores, color='k', marker='o', markerfacecolor='lightskyblue', markersize=6, label='Teste')
            plt.title("kNN Grid Search CV - Precisão vs k [Treino|Teste] (CV=10, dist=" + m + ", weights=" + w + ")")
            plt.legend()
            plt.xlabel('k')
            plt.ylabel('Precisão')
            plt.show()
            
            input(fg('red') + "\n---> Pressionar tecla aleatória para continuar <---" + fg('white'))
        




from sklearn.neural_network import MLPClassifier
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# ARTIFICIAL NEURAL NETWORKS
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
if ANN_sel:
    print("***********************************************************")
    print("*                ARTIFICIAL NEURAL NETWORKS                *")
    print("***********************************************************")
    
    warnings.filterwarnings('ignore')
    
    layer_label   = ['(2,2)', '(3,3)', '(4,4)', '(5,5)', '(6,6)', '(7,7)', '(8,8)', '(9,9)',
                     '(2,2,2)', '(3,3,3)', '(4,4,4)', '(5,5,5)', '(6,6,6)', '(7,7,7)', '(8,8,8)', '(9,9,9)']
    
    layer_sel     = [(2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9),
                     (2,2,2), (3,3,3), (4,4,4), (5,5,5), (6,6,6), (7,7,7), (8,8,8), (9,9,9)]
    func_sel      = ['identity', 'logistic', 'tanh', 'relu']
    solver_sel    = ['lbfgs', 'sgd', 'adam']
    
    for m in solver_sel:
        for w in func_sel:
            print(fg(13) + "\n##### SOLVER=" + m.upper() + " #####" + fg('white'))
            train_scores = []
            val_scores = []
            grid_train_scores = []
            grid_test_scores = []
            cv_res = []
            for idx, n in enumerate(layer_sel):
                print(fg(13) + "\n##### FUNC. ATIVATION=" + w.upper() + ", LAYER CONFIG=" + str(n) + " #####" + fg('white'))
                
                # Treino do modelo
                mdl = MLPClassifier(hidden_layer_sizes=n, activation=w, solver=m, verbose=False, early_stopping=True, validation_fraction=0.16, max_iter=5000) 
                mdl.fit(trainval_data, trainval_labels)
        
                # Predição dos dados de treino
                pred_trainval_label = mdl.predict(trainval_data)
                print(fg(11) + "--- DADOS DE TREINO (80%) ---" + fg('white'))
                cm = confusion_matrix(trainval_labels, pred_trainval_label)
                fig = plt.figure(figsize=(25,30))       
                ax = fig.add_subplot(421)
                ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
         
                plt.xlabel('Predições', fontsize=18)
                plt.ylabel('Atuais', fontsize=18)                  
                plt.title("ANN Matriz de confusão Treino (layer_config=" + str(n) + ", solver=" + m + ", act_func=" + w + ")")
    
                print(fg('light_gray') + "SCORE DE PRECISÃO: " + fg('white'), end='')
                print (accuracy_score(trainval_labels, pred_trainval_label))
                print(fg('light_gray') + "RELATÓRIO DE CLASSIFICAÇÃO:" + fg('white'))
                print (classification_report(trainval_labels, pred_trainval_label), end='')
        
                # Predição dos dados de teste
                pred_test_label = mdl.predict(test_data)
                print(fg(11) + "\n--- DADOS DE TESTE (20%) ---" + fg('white'))
                cm = confusion_matrix(test_labels, pred_test_label)   
                ax = fig.add_subplot(423)
                ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
         
                plt.xlabel('Predições', fontsize=18)
                plt.ylabel('Atuais', fontsize=18)                  
                plt.title("ANN Matriz de confusão Teste (layer_config=" + str(n) + ", solver=" + m + ", act_func=" + w + ")")
                
                print(fg('light_gray') + "SCORE DE PRECISÃO: " + fg('white'), end='')
                print (accuracy_score(test_labels, pred_test_label))
                print(fg('light_gray') + "RELATÓRIO DE CLASSIFICAÇÃO:" + fg('white'))
                print (classification_report(test_labels, pred_test_label), end='')
                
                # Plot real vs predicted
                x_ax = range(0, trainval_labels.size)
                ax = fig.add_subplot(422)
                plt.scatter(x_ax, trainval_labels, s=15, color="blue", label="original treino")
                plt.plot(x_ax, pred_trainval_label, ls='--', lw=1.5, color="red", label="predição treino")
                plt.xlabel('Amostras', fontsize=18)                
                plt.title("ANN Original vs previsão Treino (layer_config=" + str(n) + ", solver=" + m + ", act_func=" + w + ")")
                plt.legend()
                
                x_ax = range(0, test_labels.size)
                ax = fig.add_subplot(424)
                plt.scatter(x_ax, test_labels, s=15, color="blue", label="original teste")
                plt.plot(x_ax, pred_test_label, ls='--', lw=1.5, color="red", label="predição teste")
                plt.xlabel('Amostras', fontsize=18)                
                plt.title("ANN Original vs previsão Teste (layer_config=" + str(n) + ", solver=" + m + ", act_func=" + w + ")")
                plt.legend()
                plt.show()
                
                # Cálculo dos erros
                print(fg(11) + "\n--- MÉTRICAS DE ERRO ---" + fg('white'))
                mse = metrics.mean_squared_error(trainval_labels, pred_trainval_label)
                print(fg('light_gray') + "MSE TREINO: " + fg('white') + str(mse))
                rmse = math.sqrt(mse)
                print(fg('light_gray') + "RMSE TREINO: " + fg('white') + str(rmse))
                mse = metrics.mean_squared_error(test_labels, pred_test_label)
                print(fg('light_gray') + "MSE TESTE: " + fg('white') + str(mse))
                rmse = math.sqrt(mse)
                print(fg('light_gray') + "RMSE TESTE: " + fg('white') + str(rmse))
                
                # Atributos da ANN
                weights = mdl.coefs_
                biases  = mdl.intercepts_ 
                probs_trainval = mdl.predict_proba(trainval_data)
                probs_test  = mdl.predict_proba(test_data)
                
                # Avaliação (3-Fold Split)
                print(fg(11) + "\n--- AVALIAÇÃO 3-FOLD SPLIT ---" + fg('white'))              
                mdl = MLPClassifier(hidden_layer_sizes=n, activation=w, solver=m, verbose=False, early_stopping=True, validation_fraction=0.16, max_iter=5000) 
                mdl.fit(train_data, train_labels)
                
                train_scores.append(mdl.score(train_data, train_labels))
                val_scores.append(mdl.score(validation_data, validation_labels))
                print(fg('light_gray') + "SCORE DE VALIDAÇÃO: " + fg('white') + str(mdl.score(validation_data, validation_labels)))
                
                # Avaliação K-Fold CV
                print(fg(11) + "\n--- AVALIAÇÃO K-FOLD CROSS VALIDATION ---" + fg('white'))
                res = cross_validate(mdl, trainval_data, trainval_labels, return_train_score=True, cv=5)
                cv_res.append(res)
                print(fg('light_gray') + "SCORE VALIDAÇÃO CRUZADA (CV=5): " + fg('white') + str(np.mean(res['test_score'])))
                
                # Avaliação Grid Search CV
                print(fg(11) + "\n--- AVALIAÇÃO GRID SEARCH CV ---" + fg('white'))
                parameters = {'batch_size': [16, 32, 64, 128], 'solver': [m], 'activation': [w],
                              'hidden_layer_sizes': [n]}
                grid = GridSearchCV(mdl, param_grid=parameters, cv=10, return_train_score=True)
                grid.fit(trainval_data, trainval_labels)
                
                grid_train_scores.append(grid.score(trainval_data, trainval_labels))
                grid_test_scores.append(grid.score(test_data, test_labels))
                print(fg('light_gray') + "SCORE TESTE GRID SEARCH CV (CV=10): " + fg('white') + str(grid.score(test_data, test_labels)))
                
                if idx == 10:
                    input(fg('red') + "\n---> Pressionar tecla aleatória para continuar os próximos 10 cálculos <---" + fg('white'))
                
            print(fg(13) + "\n##### SOLVER=" + m.upper() +  ", FUNC. ATIVATION=" + w.upper() + " #####" + fg('white'), end='')
            
            # 3-Fold Split
            print(fg(11) + "\n--- RESULTADOS 3-FOLD SPLIT - TREINO/VALIDAÇÃO/TEST (64%/16%/20%) ---" + fg('white'))
            print(fg('light_gray') + "-> MELHOR SCORE DE VALIDAÇÃO: " + fg('white') + str(np.max(val_scores)))
            best_layer = layer_sel[np.argmax(val_scores)]
            print(fg('light_gray') + "-> MELHOR CONFIGURAÇÃO DE CAMADAS: " + fg('white') + str(best_layer))
            mdl = MLPClassifier(hidden_layer_sizes=best_layer, activation=w, solver=m, verbose=False, early_stopping=True, validation_fraction=0.16, max_iter=5000) 
            mdl.fit(trainval_data, trainval_labels)
            
            print(fg('light_gray') + "-> MELHOR SCORE DE TESTE:     " + fg('white') + str(mdl.score(test_data, test_labels)))
            
            fig = plt.figure(figsize=(15,15))         
            ax = fig.add_subplot(311)
            ax.plot(layer_label, train_scores, color='k', marker='o', markerfacecolor='royalblue', markersize=6, label='Treino')
            ax.plot(layer_label, val_scores, color='k', marker='o', markerfacecolor='lightskyblue', markersize=6, label='Validação')
            plt.title("ANN 3-Fold Split - Precisãovs vs Config. Camada [Treino|Validação] (solver=" + m + ", act_func=" + w + ")")
            plt.legend()
            plt.xlabel('Configuração de camadas/percetrões')
            plt.ylabel('Precisão')
            
             
            # K-Fold CV
            print(fg(11) + "\n--- RESULTADOS K-FOLD CROSS VALIDATION (CV) ---" + fg('white'))
            cv_mean = []
            for idx, i in enumerate(cv_res):
                cv_mean.append(np.mean(cv_res[idx]['test_score']))
                
            print(fg('light_gray') + "-> MELHOR SCORE VALIDAÇÃO CRUZADA (CV=5): " + fg('white') + str(np.max(cv_mean)))
            for idx, i in enumerate(cv_mean):
                if cv_mean[idx] == np.max(cv_mean):
                    best_layer = idx + 1
                    break
            print(fg('light_gray') + "-> MELHOR CONFIGURAÇÃO DE CAMADAS: " + fg('white') + str(layer_sel[best_layer-1]))
                 
            ax = fig.add_subplot(312)
            ax.plot(layer_label, train_scores, color='k', marker='o', markerfacecolor='royalblue', markersize=6, label='Treino')
            ax.plot(layer_label, cv_mean, color='k', marker='o', markerfacecolor='lightskyblue', markersize=6, label='Validação CV')
            plt.title("ANN K-Fold CV (CV=5, solver=" + m + ", act_func=" + w + ")")
            plt.legend()
            plt.xlabel('Configuração de camadas/percetrões')
            plt.ylabel('Precisão')
            
            
            # Grid Search CV
            mdl = MLPClassifier(activation=w, solver=m, verbose=False, early_stopping=True, validation_fraction=0.16, max_iter=5000)
            parameters = {'batch_size': [16, 32, 64, 128], 'solver': [m], 'activation': [w],
                          'hidden_layer_sizes': layer_sel}
            grid = GridSearchCV(mdl, param_grid=parameters, cv=10, return_train_score=True)
            grid.fit(trainval_data, trainval_labels)
            
            print(fg(11) + "\n--- RESULTADOS GRID SEARCH CV ---" + fg('white'))
            print(fg('light_gray') + "-> MELHOR SCORE DE VALIDAÇÃO CRUZADA (CV=10): " + fg('white') + str(grid.best_score_))
            print(fg('light_gray') + "-> MELHOR CONFIGURAÇÃO: " + fg('white') + str(grid.best_params_))
            print(fg('light_gray') + "-> MELHOR SCORE DE TESTE:     " + fg('white') + str(grid.score(test_data, test_labels)))
            
            ax = fig.add_subplot(313)
            ax.plot(layer_label, grid_train_scores, color='k', marker='o', markerfacecolor='royalblue', markersize=6, label='Treino')
            ax.plot(layer_label, grid_test_scores, color='k', marker='o', markerfacecolor='lightskyblue', markersize=6, label='Teste')
            plt.title("ANN Grid Search CV - Precisão vs Config. Camada [Treino|Teste] (CV=10, solver=" + m + ", act_func=" + w + ")")
            plt.legend()
            plt.xlabel('Configuração de camadas/percetrões')
            plt.ylabel('Precisão')
            plt.show()
            
            input(fg('red') + "\n---> Pressionar tecla aleatória para continuar <---" + fg('white'))





from sklearn.svm import SVC
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# SUPPORT VECTOR MACHINES
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
if SVM_sel:
    print("***********************************************************")
    print("*                 SUPPORT VECTOR MACHINES                 *")
    print("***********************************************************")
    
    warnings.filterwarnings('ignore')
    
    reg_sel     = [x / 10.0 for x in range(5, 100, 5)]
    kernel_sel  = ['linear', 'poly', 'rbf', 'sigmoid']
    
    for w in kernel_sel:
        train_scores = []
        val_scores = []
        grid_train_scores = []
        grid_test_scores = []
        cv_res = []
        for idx, n in enumerate(reg_sel):
            print(fg(13) + "\n##### KERNEL=" + w.upper() + ", REGULARIZATION=" + str(n) + " #####" + fg('white'))
            
            # Treino do modelo
            clf = SVC(C=n, kernel=w, max_iter=5000)
            clf.fit(trainval_data, trainval_labels)
    
            # Predição dos dados de treino
            pred_trainval_label = clf.predict(trainval_data)
            print(fg(11) + "--- DADOS DE TREINO (80%) ---" + fg('white'))
            cm = confusion_matrix(trainval_labels, pred_trainval_label)
            fig = plt.figure(figsize=(25,30))       
            ax = fig.add_subplot(421)
            ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
     
            plt.xlabel('Predições', fontsize=18)
            plt.ylabel('Atuais', fontsize=18)                  
            plt.title("SVM Matriz de confusão Treino (regularization=" + str(n) + ", kernel=" + w + ")")
            plt.grid(False)

            print(fg('light_gray') + "SCORE DE PRECISÃO: " + fg('white'), end='')
            print (accuracy_score(trainval_labels, pred_trainval_label))
            print(fg('light_gray') + "RELATÓRIO DE CLASSIFICAÇÃO:" + fg('white'))
            print (classification_report(trainval_labels, pred_trainval_label), end='')
    
            # Predição dos dados de teste
            pred_test_label = clf.predict(test_data)
            print(fg(11) + "\n--- DADOS DE TESTE (20%) ---" + fg('white'))
            cm = confusion_matrix(test_labels, pred_test_label)   
            ax = fig.add_subplot(423)
            ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
     
            plt.xlabel('Predições', fontsize=18)
            plt.ylabel('Atuais', fontsize=18)                  
            plt.title("SVM Matriz de confusão Teste (regularization=" + str(n) + ", kernel=" + w + ")")
            plt.grid(False)
            
            print(fg('light_gray') + "SCORE DE PRECISÃO: " + fg('white'), end='')
            print (accuracy_score(test_labels, pred_test_label))
            print(fg('light_gray') + "RELATÓRIO DE CLASSIFICAÇÃO:" + fg('white'))
            print (classification_report(test_labels, pred_test_label), end='')
            
            # Plot real vs predicted
            x_ax = range(0, trainval_labels.size)
            ax = fig.add_subplot(422)
            plt.scatter(x_ax, trainval_labels, s=15, color="blue", label="original treino")
            plt.plot(x_ax, pred_trainval_label, ls='--', lw=1.5, color="red", label="predição treino")
            plt.xlabel('Amostras', fontsize=18)                
            plt.title("SVM Original vs previsão Treino (regularization=" + str(n) + ", kernel=" + w + ")")
            plt.grid(False)
            plt.legend()
            
            x_ax = range(0, test_labels.size)
            ax = fig.add_subplot(424)
            plt.scatter(x_ax, test_labels, s=15, color="blue", label="original teste")
            plt.plot(x_ax, pred_test_label, ls='--', lw=1.5, color="red", label="predição teste")
            plt.xlabel('Amostras', fontsize=18)                
            plt.title("SVM Original vs previsão Teste (regularization=" + str(n) + ", kernel=" + w + ")")
            plt.grid(False)
            plt.legend()
            plt.show()
            
            # Cálculo dos erros
            print(fg(11) + "\n--- MÉTRICAS DE ERRO ---" + fg('white'))
            mse = metrics.mean_squared_error(trainval_labels, pred_trainval_label)
            print(fg('light_gray') + "MSE TREINO: " + fg('white') + str(mse))
            rmse = math.sqrt(mse)
            print(fg('light_gray') + "RMSE TREINO: " + fg('white') + str(rmse))
            mse = metrics.mean_squared_error(test_labels, pred_test_label)
            print(fg('light_gray') + "MSE TESTE: " + fg('white') + str(mse))
            rmse = math.sqrt(mse)
            print(fg('light_gray') + "RMSE TESTE: " + fg('white') + str(rmse))
            
            
            # Avaliação (3-Fold Split)
            print(fg(11) + "\n--- AVALIAÇÃO 3-FOLD SPLIT ---" + fg('white'))      
            clf = SVC(C=n, kernel=w, max_iter=5000)
            clf.fit(train_data, train_labels)
            
            train_scores.append(clf.score(train_data, train_labels))
            val_scores.append(clf.score(validation_data, validation_labels))
            print(fg('light_gray') + "SCORE DE VALIDAÇÃO: " + fg('white') + str(clf.score(validation_data, validation_labels)))
            
            # Avaliação K-Fold CV
            print(fg(11) + "\n--- AVALIAÇÃO K-FOLD CROSS VALIDATION ---" + fg('white'))
            res = cross_validate(clf, trainval_data, trainval_labels, return_train_score=True, cv=5)
            cv_res.append(res)
            print(fg('light_gray') + "SCORE VALIDAÇÃO CRUZADA (CV=5): " + fg('white') + str(np.mean(res['test_score'])))
            
            # Avaliação Grid Search CV
            print(fg(11) + "\n--- AVALIAÇÃO GRID SEARCH CV ---" + fg('white'))
            parameters = {'C': [n], 'kernel': [w]}
            grid = GridSearchCV(clf, param_grid=parameters, cv=10, return_train_score=True)
            grid.fit(trainval_data, trainval_labels)
            
            grid_train_scores.append(grid.score(trainval_data, trainval_labels))
            grid_test_scores.append(grid.score(test_data, test_labels))
            print(fg('light_gray') + "SCORE TESTE GRID SEARCH CV (CV=10): " + fg('white') + str(grid.score(test_data, test_labels)))
            
            if n == 5:
                input(fg('red') + "\n---> Pressionar tecla aleatória para continuar os próximos 10 cálculos <---" + fg('white'))
            
        print(fg(13) + "\n##### KERNEL=" + w.upper() + " #####" + fg('white'), end='')
        
        # 3-Fold Split
        print(fg(11) + "\n--- RESULTADOS 3-FOLD SPLIT - TREINO/VALIDAÇÃO/TEST (64%/16%/20%) ---" + fg('white'))
        print(fg('light_gray') + "-> MELHOR SCORE DE VALIDAÇÃO: " + fg('white') + str(np.max(val_scores)))
        best_reg = reg_sel[np.argmax(val_scores)]
        print(fg('light_gray') + "-> MELHOR REGULARIZAÇÃO: " + fg('white') + str(best_reg))
        clf = SVC(C=best_reg, kernel=w, max_iter=5000)
        clf.fit(trainval_data, trainval_labels)
        
        print(fg('light_gray') + "-> MELHOR SCORE DE TESTE:     " + fg('white') + str(clf.score(test_data, test_labels)))
        
        fig = plt.figure(figsize=(15,15))         
        ax = fig.add_subplot(311)
        ax.plot(reg_sel, train_scores, color='k', marker='o', markerfacecolor='royalblue', markersize=6, label='Treino')
        ax.plot(reg_sel, val_scores, color='k', marker='o', markerfacecolor='lightskyblue', markersize=6, label='Validação')
        plt.title("SVM 3-Fold Split - Precisão vs Regularização [Treino|Validação] (kernel=" + w + ")")
        plt.grid(False)
        plt.legend()
        plt.xlabel('Regularização')
        plt.ylabel('Precisão')
        
         
        # K-Fold CV
        print(fg(11) + "\n--- RESULTADOS K-FOLD CROSS VALIDATION (CV) ---" + fg('white'))
        cv_mean = []
        for idx, i in enumerate(cv_res):
            cv_mean.append(np.mean(cv_res[idx]['test_score']))
            
        print(fg('light_gray') + "-> MELHOR SCORE VALIDAÇÃO CRUZADA (CV=5): " + fg('white') + str(np.max(cv_mean)))
        for idx, i in enumerate(cv_mean):
            if cv_mean[idx] == np.max(cv_mean):
                best_reg = idx + 1
                break
        print(fg('light_gray') + "-> MELHOR REGULARIZAÇÃO: " + fg('white') + str(reg_sel[best_reg-1]))
             
        ax = fig.add_subplot(312)
        ax.plot(reg_sel, train_scores, color='k', marker='o', markerfacecolor='royalblue', markersize=6, label='Treino')
        ax.plot(reg_sel, cv_mean, color='k', marker='o', markerfacecolor='lightskyblue', markersize=6, label='Validação CV')
        plt.title("SVM K-Fold CV (CV=5, kernel=" + w + ")")
        plt.grid(False)
        plt.legend()
        plt.xlabel('Regularização')
        plt.ylabel('Precisão')
        
        
        # Grid Search CV
        clf = SVC(C=best_reg, kernel=w, max_iter=5000)
        parameters = {'C': [n], 'kernel': [w]}
        grid = GridSearchCV(clf, param_grid=parameters, cv=10, return_train_score=True)
        grid.fit(trainval_data, trainval_labels)
        
        print(fg(11) + "\n--- RESULTADOS GRID SEARCH CV ---" + fg('white'))
        print(fg('light_gray') + "-> MELHOR SCORE DE VALIDAÇÃO CRUZADA (CV=10): " + fg('white') + str(grid.best_score_))
        print(fg('light_gray') + "-> MELHOR CONFIGURAÇÃO: " + fg('white') + str(grid.best_params_))
        print(fg('light_gray') + "-> MELHOR SCORE DE TESTE:     " + fg('white') + str(grid.score(test_data, test_labels)))
        
        ax = fig.add_subplot(313)
        ax.plot(reg_sel, grid_train_scores, color='k', marker='o', markerfacecolor='royalblue', markersize=6, label='Treino')
        ax.plot(reg_sel, grid_test_scores, color='k', marker='o', markerfacecolor='lightskyblue', markersize=6, label='Teste')
        plt.title("SVM Grid Search CV - Precisão vs Regularização [Treino|Teste] (CV=10, kernel=" + w + ")")
        plt.grid(False)
        plt.legend()
        plt.xlabel('Regularização')
        plt.ylabel('Precisão')
        plt.show()
        
        input(fg('red') + "\n---> Pressionar tecla aleatória para continuar <---" + fg('white'))

