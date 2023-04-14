class npArr:
    """ 
    DESC: Função que recebe um NumPy Array 'df', verifica as colunas indicadas pelo range 'rg' e substitui os valores NaN pela média 
          da respetiva coluna. Devolve o array já corrigido.
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
    DESC: Função que recebe um NumPy Array 'df', verifica as colunas indicadas pelo range 'rg' e aplica normalização standard ao array.
          Devolve o NumPy Array já normalizado.
          -------------------------------------------------------------------
          (IN):   df      - NumPy Array
                  rg      - range() das colunas de df a analisar
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
Função de Jörn's alterada 
"""
def fancy_dendrogram(*args, **kwargs):
    plt.rcParams['lines.linewidth'] = 4
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', None)

    ddata = dendrogram(*args, **kwargs)
    plt.grid(False)

    if not kwargs.get('no_plot', False):
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" %y, (x,y), xytext=(15,0), textcoords='offset points', va='top', ha='center')
        if max_d:
            plt.rcParams['lines.linewidth'] = 1
            plt.axhline(y=max_d, c='k', ls='--')
            
    return ddata



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# INÍCIO DO CÓDIGO
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
h_cluster_check = True
kmeans_check    = True
fuzz_check      = True


import numpy as np 
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pyautogui
import time

# Apaga os plots guardados no Spyder. «!» Só funciona se a respetiva janela 'Plots' 
# estiver aberta e selecionada no canto superior direito
pyautogui.moveTo(1500, 350)                 # Movimenta cursor para X e Y
pyautogui.click()                           # Mouse click
with pyautogui.hold('ctrl'):                # pressiona Ctrl+Shift+W
    with pyautogui.hold('shit'):    
        pyautogui.press('W')

dataset = pd.read_excel('Dataset.xlsx', sheet_name='Clustering')
ds_size = dataset.size
dataset = np.array(dataset)
dataset = dataset[:, 0:4]



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# PREPARAÇÃO DO DATASET
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

#---------------------------------------------
# CORREÇÃO DO DATASET
dataset = npArr.NaN2Mean(dataset, range(1,4)) # Correção de valores NaN para média da respetiva coluna

#---------------------------------------------
# PLOTS 
labels = range(1, ds_size) 
plt.rcParams['lines.linewidth'] = 1

# Plot da feature Tenure
fig1 = plt.figure(figsize=(10,10))
ax = fig1.add_subplot(211)      
plt.title("Feature Tenure")                
plt.xlabel("ID")
plt.ylabel("Tenure")
ax.scatter(dataset[:,0], dataset[:,1], label='True Position')
plt.grid(True) 
for label, x, y in zip(labels, dataset[:,0], dataset[:,1]):   
    plt.annotate(label, xy=(x,y), xytext=(-3,3), textcoords='offset points', ha='right', va='bottom')

ax = fig1.add_subplot(212)   
plt.title("Feature Tenure")                
plt.xlabel("Tenure")
ax.boxplot(dataset[:,1], vert=False)
plt.grid(False)
plt.show() 
    
# Plot da feature MonthlyCharges
fig2 = plt.figure(figsize=(10,10))
ax = fig2.add_subplot(211)         
plt.title("Feature MonthlyCharges")          
plt.xlabel("ID")
plt.ylabel("MonthlyCharges")
ax.scatter(dataset[:,0], dataset[:,2], label='True Position') 
plt.grid(True) 
for label, x, y in zip(labels, dataset[:,0], dataset[:,2]):
    plt.annotate(label, xy=(x,y), xytext=(-3,3), textcoords='offset points', ha='right', va='bottom')

ax = fig2.add_subplot(212)   
plt.title("Feature MonthlyCharges")                     
plt.xlabel("MonthlyCharges")
ax.boxplot(dataset[:,2], vert=False)
plt.grid(False)
plt.show() 

# Plot da feature TotalCharges
fig3 = plt.figure(figsize=(10,10))
ax = fig3.add_subplot(211)          
plt.title("Feature TotalCharges")               
plt.xlabel("ID")
plt.ylabel("TotalCharges")
ax.scatter(dataset[:,0], dataset[:,3], label='True Position')
plt.grid(True) 
for label, x, y in zip(labels, dataset[:,0], dataset[:,3]): 
    plt.annotate(label, xy=(x,y), xytext=(-3,3), textcoords='offset points', ha='right', va='bottom')
      
ax = fig3.add_subplot(212)       
plt.title("Feature TotalCharges")             
plt.xlabel("TotalCharges")
ax.boxplot(dataset[:,3], vert=False)
plt.grid(False)
plt.show()
print("\n\n\n\n\n")

#---------------------------------------------
# NORMALIZAÇÃO
dataset_n = npArr.std_normalization(dataset, range(1,4))

# Extração das features
features = dataset_n[:,1:4]
xlim = []
xlim.append(np.min(dataset_n[:,0]))
xlim.append(np.max(dataset_n[:,0]))
ylim = []
ylim.append(np.min(dataset_n[:,1]))
ylim.append(np.max(dataset_n[:,1]))



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# CLUSTERING HIERÁRQUICO
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster 
from kneed import KneeLocator
import warnings

# Dendrogramas + scatter plots
#dist_sel = ['euclidean']
#linkage_sel = ['average']
dist_sel    = ['canberra', 'chebyshev', 'cityblock', 'cosine', 'euclidean', 'sqeuclidean']
linkage_sel = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']

if h_cluster_check:
    print("   --- CLUSTERING HIERÁRQUICO ---\n")
    for i in dist_sel:
        for k in linkage_sel: 
            time_start = round(time.time() * 1000)
            if (k == 'ward' or k == 'centroid' or k == 'median') and not i == 'euclidean':
                continue
            
            Z = linkage(features, method=k, metric=i) # tipo de ligação
            max_d = 0.59*max(Z[:,2])                  # corte a 59%
            
            cmap = plt.cm.gist_rainbow(np.linspace(0, 1, 10))
            hierarchy.set_link_color_palette([matplotlib.colors.rgb2hex(rgb[:3]) for rgb in cmap])
            
            plt.figure(figsize=(20,10)).add_subplot(111)      
            d = fancy_dendrogram(Z, leaf_rotation=90., leaf_font_size=12., annotate_above=10, show_contracted=True, 
                                 max_d=max_d, above_threshold_color='k')
            plt.grid(False)
            s = "Hierarchical Clustering Dendrogram (dist=" + i + ", linkage=" + k + ")"
            plt.title(s)
            plt.xlabel('Índice da amostra/Tamanho do cluster')
            plt.ylabel('Distância')
                     
            fig = plt.figure(figsize=(10,10))
            fig.add_subplot(111)      
            plt.scatter(features[d['leaves'],0], features[d['leaves'],1], color=d['leaves_color_list'])  # plot Tenure por MonthlyCharges
            plt.grid(False)
            s = "Scatter Plot (dist=" + i + ", linkage=" + k + ")"
            plt.title(s)  
            plt.xlabel("Tenure")
            plt.ylabel("MonthlyCharges")
            
            
            # Método do cotovelo (soma dos quadrados)   
            plt.figure(figsize=(10,10)).add_subplot(111)      
            last = Z[-10:,2]
            last_rev = last[::-1]
            idxs = np.arange(1, len(last) + 1)
            s = "Método do Cotovelo (dist=" + i + ", linkage=" + k + ")"
            plt.title(s)
            plt.xlabel('k')
            plt.ylabel('Distância')
            plt.plot(idxs, last_rev, color='b', marker='o', ls='-', label="curva de distâncias")
            plt.grid(False)               
            
            # localizador de joelho
            warnings.filterwarnings("ignore")
            try:
                n = KneeLocator(idxs[1:idxs.size], last_rev[1:last_rev.size], curve='convex', direction='decreasing', online=False, S=0.5)
            except RuntimeWarning:
                n = []
                n.knee = 'None'
            denorm_x = n.x_normalized * (max(idxs[1:idxs.size]) - min(idxs[1:idxs.size])) + min(idxs[1:idxs.size])
            denorm_y = n.y_normalized * (max(last_rev[1:last_rev.size]) - min(last_rev[1:last_rev.size])) + min(last_rev[1:last_rev.size])
            plt.plot(denorm_x, denorm_y, color='r', ls='-', label="curva de deteção")
            
            if isinstance(n.knee, (int, np.int32)):
                plt.axvline(x=n.knee, c='k', ls='--', label="joelho")
                plt.grid(False) 
                temp = n.knee
            else:
                temp = 'No'
            plt.legend(loc="upper left", ncol=5, fontsize=12, facecolor="white", edgecolor="black",
                       frameon=True, framealpha=0.5) 
            plt.show()
            
            s = "   Método do Cotovelo (dist=" + i
            if i == 'cosine':
                s = s + ",     "
            elif i == 'chebyshev' or i == 'cityblock' or i == 'euclidean':
                s = s + ",  "
            elif i == 'canberra':
                s = s + ",   "    
            else:
                s = s + ","
            s = s + " linkage=" + k + ")" 
            if k == 'ward':
                s = s + "\t\t"
            else:
                s = s + "\t"
            s = s +"k ótimo: " +  str(temp)
            time_end = round(time.time() * 1000)
            s = s + "\t [Tempo de iteração = " + str(time_end - time_start) + " ms]"
            print(s)
    print("\n")



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# MÉTODO K-MEANS
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
from skcmeans.algorithms import Probabilistic, Hard
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.metrics import silhouette_score
import decimal

if kmeans_check:
    # Scatter plots 
    n_clusters = range(2,11)
    for k in n_clusters:
        plt.figure(figsize=(10,7)).add_subplot(111)   
        km = KMeans(n_clusters=k, init='random', random_state=42, max_iter=5000, n_init=100).fit(features)
        C = fcluster(Z, k, criterion='maxclust')
        centroids = km.cluster_centers_
        
        plt.scatter(centroids[:,0], centroids[:,1], c="k", marker='x')           # plot centroides
        
        for idx in range(0, k):
            plt.scatter(centroids[idx,0], centroids[idx,1], c='grey', marker='X', edgecolors='k', 
                        linewidths=1.5, label=("C" + str(idx)), s=75)            # centroides de cada cluster         
        
        plt.scatter(features[:,0], features[:,1], c=C, cmap='gist_rainbow')      # plot Tenure por MonthlyCharges
        plt.grid(False)
        plt.xlabel("Tenure")
        plt.ylabel("MonthlyCharges")
        s ="Scatter Plot K-Means (Hard Clustering), dist=euclidean, k=" + str(k)
        plt.title(s)  
        plt.show()
    
    print("   --- K-MEANS [1ª abordagem] ---\n")
    # Método da silhueta
    silhouette_avg = []
    for idx, x in enumerate(n_clusters):
        km = KMeans(n_clusters=x, init='random', random_state=42, max_iter=5000, n_init=100).fit(features)
        silhouette_avg.append(silhouette_score(features, km.labels_))
        print("   Pontuação média Método da Silhueta: " + str(decimal.Decimal(silhouette_avg[idx]).quantize(decimal.Decimal('0.00000'))) 
              + "\t(dist=euclidean, k = " + str(x) + ")")
    
    for idx, x in enumerate(n_clusters):
        if silhouette_avg[idx] == max(silhouette_avg):
            print("      -> Método da Silhueta (dist=euclidean) k ótimo: " + str(x), end="\n")
    
    for k in n_clusters:
        km = KMeans(n_clusters=k, init='random', random_state=42, max_iter=5000, n_init=100).fit(features)
        
        cmap = plt.cm.gist_rainbow(np.linspace(0, 1, k))
        colors = (matplotlib.colors.rgb2hex(rgb[:3]) for rgb in cmap)
        
        plt.figure(figsize=(10,7)).add_subplot(111)   
        sil_visualizer = SilhouetteVisualizer(km, colors=colors)
        sil_visualizer.fit(features)    
        plt.grid(False)  
        sil_visualizer.show()
        
    # Método do cotovelo
    km = KMeans(init='random', random_state=42, max_iter=5000, n_init=100).fit(features)
    
    plt.figure(figsize=(10,7)).add_subplot(111)        
    elb_visualizer = KElbowVisualizer(km, k=(2,11))
    elb_visualizer.fit(features)    
    plt.grid(False)  
    elb_visualizer.show()  
    print("   Método do Cotovelo (dist=euclidean) k ótimo:", str(4), end="\n\n\n")



    print("   --- HARD C-MEANS (K-MEANS) [2ª abordagem] ---\n")
    for r in dist_sel:
        distortions = []
        silhouette_avg = []
        
        for idx_top, k in enumerate(n_clusters):
            time_start = round(time.time() * 1000)
            # Scatter plots + soft lines (Probabilistic Fuzzy C-Means)
            fig = plt.figure(figsize=(10,10))
            fig.add_subplot(111)   
            data = features[:,0:2]
            data[np.argsort(data[:, 0])]
            class IteratorKmeans(Hard):
                metric = r
            
            clusterer = IteratorKmeans(n_clusters=k, n_init=100, max_iter=5000, random_state=42)
            clusterer.fit(data)
            cntr = clusterer.calculate_centers(data)
            
            xx, yy = np.array(np.meshgrid(np.linspace(xlim[0]-0.1, xlim[1]+0.1, 1000), np.linspace(ylim[0]-0.1, xlim[1]+0.1, 1000)))
            z = np.rollaxis(clusterer.calculate_memberships(np.c_[xx.ravel(), yy.ravel()]).reshape(*xx.shape, -1), 2, 0)
              
            cmap = plt.cm.gist_rainbow(np.linspace(0, 1, k))
            colors = []
            for rgb in cmap:
                colors.append(matplotlib.colors.rgb2hex(rgb[:3]))
                
            for membership, color in zip(z, colors):
                plt.contour(xx, yy, membership, colors=color, alpha=0.5)                    # cálculo da membership para linhas
            plt.grid(False)
            
            c = clusterer.memberships                                                       # cálculo da membership para pontos das features
            color_mb = []       
            for i in range(0, c[:,0].size):
                for j in range(0, c[0,:].size):
                    if c[i,j] == np.max(c[i,:]):
                        color_mb.append(j)
            
            for i in range(0,data[:,0].size):  
                plt.scatter(data[i, 0], data[i, 1], c=colors[color_mb[i]])                  # pontos em cada cluster
            plt.grid(False)
            
            for idx, pt in enumerate(cntr):
                plt.scatter(pt[0], pt[1], c=colors[idx], marker='X', edgecolors='k', 
                            linewidths=1.5, label=("C" + str(idx)), s=75)                   # centroides de cada cluster
            plt.grid(False)
            plt.legend(loc="upper left", ncol=5, fontsize=8, facecolor="white", edgecolor="black",
                       frameon=True, framealpha=0.5) 
    
            plt.xlabel("Tenure")
            plt.ylabel("MonthlyCharges")
            s = "Scatter Plot Hard C-Means - K-Means (Hard Clustering), k=" + str(k) + ", dist=" + r
            plt.title(s)  
            
            # Método do cotovelo
            data = features[:,0:2]
            distortions.append(sum(np.min(cdist(data, cntr, r), axis=1)) / data[:,0].size)
    
            plt.show()
            
            # Método da silhueta
            silhouette_avg.append(silhouette_score(features, np.array(color_mb)))
            print("   Pontuação média Método da Silhueta: " + str(decimal.Decimal(silhouette_avg[idx_top]).quantize(decimal.Decimal('0.00000'))) 
                  + "\t(dist=" + r + ", k = " + str(k) + ")", end='')
            
            time_end = round(time.time() * 1000)
            print("\t [Tempo de iteração = " + str(time_end - time_start) + " ms]")
            
            if k == max(list(n_clusters)):
                t = range(2,11)
                for idx_x, x in enumerate(t):
                    if silhouette_avg[idx_x] == max(silhouette_avg):
                        print("      -> Método da Silhueta (dist=" + r + ") k ótimo: " + str(x), end='\n')
        
        fig = plt.figure(figsize=(10,10))
        fig.add_subplot(111)   
        last = np.array(distortions[::-1])
        last_rev = last[::-1]
        idxs = np.arange(1, len(last) + 1)
        s = "Método do Cotovelo (dist=" + r + ")"
        plt.title(s)
        plt.xlabel('k')
        plt.ylabel('Distância')
        plt.plot(idxs, last_rev, color='b', marker='o', ls='-', label="curva de distâncias")
        plt.grid(False)               
        
        # localizador de joelho
        warnings.filterwarnings("ignore")
        try:
            n = KneeLocator(idxs[1:idxs.size], last_rev[1:last_rev.size], curve='convex', direction='decreasing', online=False, S=0.5)
        except RuntimeWarning:
            n = []
            n.knee = 'None'
        denorm_x = n.x_normalized * (max(idxs[1:idxs.size]) - min(idxs[1:idxs.size])) + min(idxs[1:idxs.size])
        denorm_y = n.y_normalized * (max(last_rev[1:last_rev.size]) - min(last_rev[1:last_rev.size])) + min(last_rev[1:last_rev.size])
        plt.plot(denorm_x, denorm_y, color='r', ls='-', label="curva de deteção")
        
        if isinstance(n.knee, (int, np.int32)):
            plt.axvline(x=n.knee, c='k', ls='--', label="joelho")
            plt.grid(False) 
            temp = n.knee
        else:
            temp = 'No'
        plt.legend(loc="upper left", ncol=5, fontsize=12, facecolor="white", edgecolor="black",
                   frameon=True, framealpha=0.5) 
        plt.show()
        
        s = "   Método do cotovelo c/ distorção (dist=" + r 
        if r == 'cosine':
            s = s + ")\t\t"
        elif not r == 'sqeuclidean':
            s = s + ")\t\t"
        else:
            s = s + ")\t"
        s = s +"k ótimo: " + str(n.knee)
        print(s, end='\n\n')
           
    print("\n")



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# MÉTODO “FUZZY” C-MEANS
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$    
if fuzz_check:
    n_clusters = range(2,11)
    
    print("   --- FUZZY C-MEANS ---\n") 
    for r in dist_sel:
        distortions = []
        silhouette_avg = []
        
        for idx_top, k in enumerate(n_clusters):
            time_start = round(time.time() * 1000)
            # Scatter plots + soft lines (Probabilistic Fuzzy C-Means)
            fig = plt.figure(figsize=(10,10))
            fig.add_subplot(111)   
            data = features[:,0:2]
            data[np.argsort(data[:, 0])]
            class IteratorProbabilistic(Probabilistic):
                metric = r
            
            clusterer = IteratorProbabilistic(n_clusters=k, n_init=100, max_iter=5000, random_state=42)
            clusterer.fit(data)
            cntr = clusterer.calculate_centers(data)
            
            xx, yy = np.array(np.meshgrid(np.linspace(xlim[0]-0.1, xlim[1]+0.1, 1000), np.linspace(ylim[0]-0.1, xlim[1]+0.1, 1000)))
            z = np.rollaxis(clusterer.calculate_memberships(np.c_[xx.ravel(), yy.ravel()]).reshape(*xx.shape, -1), 2, 0)
              
            cmap = plt.cm.gist_rainbow(np.linspace(0, 1, k))
            colors = []
            for rgb in cmap:
                colors.append(matplotlib.colors.rgb2hex(rgb[:3]))
                
            for membership, color in zip(z, colors):
                plt.contour(xx, yy, membership, colors=color, alpha=0.5)                    # cálculo da membership para linhas
            plt.grid(False)
            
            c = clusterer.memberships                                                       # cálculo da membership para pontos das features
            color_mb = []       
            for i in range(0, c[:,0].size):
                for j in range(0, c[0,:].size):
                    if c[i,j] == np.max(c[i,:]):
                        color_mb.append(j)
            
            for i in range(0,data[:,0].size):  
                plt.scatter(data[i, 0], data[i, 1], c=colors[color_mb[i]])                  # pontos em cada cluster
            plt.grid(False)
            
            for idx, pt in enumerate(cntr):
                plt.scatter(pt[0], pt[1], c=colors[idx], marker='X', edgecolors='k', 
                            linewidths=1.5, label=("C" + str(idx)), s=75)                   # centroides de cada cluster
            plt.grid(False)
            plt.legend(loc="upper left", ncol=5, fontsize=8, facecolor="white", edgecolor="black",
                       frameon=True, framealpha=0.5) 
    
            plt.xlabel("Tenure")
            plt.ylabel("MonthlyCharges")
            s = "Scatter Plot Fuzzy C-Means (Soft Clustering), k=" + str(k) + ", dist=" + r
            plt.title(s)  
            
            # Método do cotovelo
            data = features[:,0:2]
            distortions.append(sum(np.min(cdist(data, cntr, r), axis=1)) / data[:,0].size)
    
            plt.show()
            
            # Método da silhueta
            silhouette_avg.append(silhouette_score(features, np.array(color_mb)))
            print("   Pontuação média Método da Silhueta: " + str(decimal.Decimal(silhouette_avg[idx_top]).quantize(decimal.Decimal('0.00000'))) 
                  + "\t(dist=" + r + ", k = " + str(k) + ")", end='')
    
            time_end = round(time.time() * 1000)
            print("\t [Tempo de iteração = " + str(time_end - time_start) + " ms]")
            
            if k == max(list(n_clusters)):
                t = range(2,11)
                for idx_x, x in enumerate(t):
                    if silhouette_avg[idx_x] == max(silhouette_avg):
                        print("      -> Método da Silhueta (dist=" + r + ") k ótimo: " + str(x), end='\n')
    
        fig = plt.figure(figsize=(10,10))
        fig.add_subplot(111)   
        last = np.array(distortions[::-1])
        last_rev = last[::-1]
        idxs = np.arange(1, len(last) + 1)
        s = "Método do Cotovelo (dist=" + r + ")"
        plt.title(s)
        plt.xlabel('k')
        plt.ylabel('Distância')
        plt.plot(idxs, last_rev, color='b', marker='o', ls='-', label="curva de distâncias")
        plt.grid(False)               
        
        # localizador de joelho
        warnings.filterwarnings("ignore")
        try:
            n = KneeLocator(idxs[1:idxs.size], last_rev[1:last_rev.size], curve='convex', direction='decreasing', online=False, S=0.5)
        except RuntimeWarning:
            n = []
            n.knee = 'None'
        denorm_x = n.x_normalized * (max(idxs[1:idxs.size]) - min(idxs[1:idxs.size])) + min(idxs[1:idxs.size])
        denorm_y = n.y_normalized * (max(last_rev[1:last_rev.size]) - min(last_rev[1:last_rev.size])) + min(last_rev[1:last_rev.size])
        plt.plot(denorm_x, denorm_y, color='r', ls='-', label="curva de deteção")
        
        if isinstance(n.knee, (int, np.int32)):
            plt.axvline(x=n.knee, c='k', ls='--', label="joelho")
            plt.grid(False) 
            temp = n.knee
        else:
            temp = 'No'
        plt.legend(loc="upper left", ncol=5, fontsize=12, facecolor="white", edgecolor="black",
                   frameon=True, framealpha=0.5) 
        plt.show()
        
        s = "   Método do cotovelo c/ distorção (dist=" + r 
        if r == 'cosine':
            s = s + ")\t\t"
        elif not r == 'sqeuclidean':
            s = s + ")\t\t"
        else:
            s = s + ")\t"
        s = s +"k ótimo: " + str(n.knee)
        print(s, end='\n\n')
        
    print("\n")
