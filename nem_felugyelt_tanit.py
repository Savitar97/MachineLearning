# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 16:23:53 2020

@author: IfritR
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

wine = pd.read_csv(url, sep = ';')
wine.mean().plot(kind='bar')
wine_dropped = wine.drop('quality', axis=1)
# Normalizing over the standard deviation
X =wine_dropped.values[:, 1:]
#X=StandardScaler().fit_transform(X)
#X = wine.drop('quality', axis=1)
x_cols=wine_dropped.columns
#X2=X.to_numpy(copy=True)
y=wine['quality']
#y2=y.to_numpy(copy=True)

#box plot for features 
f2, ax = plt.subplots(2, 2, figsize=(16, 12))
sns.boxplot('quality', 'alcohol', data=wine, ax=ax[0, 0], palette='Blues')
sns.boxplot('quality', 'sulphates', data=wine, ax=ax[0, 1], palette='Blues')
sns.boxplot('quality', 'volatile acidity', data=wine, ax=ax[1, 0], palette='Blues')
sns.boxplot('quality', 'citric acid', data=wine, ax=ax[1, 1], palette='Blues')

#Plot elbow method figure
plt.figure(6);
distortions = []
for i in range (1,11):
    km = KMeans(n_clusters=i,
               max_iter=300,
               random_state=2020)
    km.fit(X)
    distortions.append(km.inertia_)
    
plt.plot(range(1,11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()


#Kmeans clustering
kmeans =KMeans(init='k-means++',n_clusters=2,random_state=2020,n_init=11)
kmeans.fit(X)
wine_labels = kmeans.labels_
wine_centers = kmeans.cluster_centers_
distX = kmeans.transform(X);
dist_center = kmeans.transform(wine_centers);
sse = kmeans.inertia_# Sum of Squared Errors(SSE)
score = kmeans.score(X)
wine_dropped['Cluster'] = wine_labels#add cluster column to data
cluster_details=wine_dropped.groupby('Cluster').mean()#means group by clusters
print(cluster_details)


# Visualizing of clustering in the distance space
fig = plt.figure(4);
plt.title('Wine data in the distance space');
plt.xlabel('Cluster 1');
plt.ylabel('Cluster 2');
plt.scatter(distX[:,0],distX[:,1],s=50,c=wine_labels);  # data
plt.scatter(dist_center[:,0],dist_center[:,1],s=10,c='red',marker='X');  # centroids
plt.show();

#Calculate davies boulding score
DB = davies_bouldin_score(X,wine_labels)

print(f'Within SSE: {sse}')
print(f'Davies-Bouldin index: {DB}')

"""
total sulfur dioxide:
amount of free and bound forms of S02; in low concentrations,
SO2 is mostly undetectable in wine, but at free SO2 
"""
#plot
fig = plt.figure(3,figsize=(12,12));
plt.xlabel('alcohol', fontsize=18)
plt.ylabel('total sulfur dioxide', fontsize=16)
plt.scatter(X[:,9],X[:,5],c=wine_labels,label=wine_labels)  # data
plt.scatter(wine_centers[:,9],wine_centers[:,5],c='red',marker='X')  # centroids
plt.show()
plt.xlabel('alcohol', fontsize=18)
plt.ylabel('total sulfur dioxide', fontsize=16)
sns.scatterplot(x=X[:,9], y=X[:,5], hue="Cluster", 
                data=wine_dropped, palette='Paired', s=20);
plt.legend(loc='lower right');


#PCA
pca = PCA(n_components=2);
pca.fit(X);
wine_pc = pca.transform(X);  #  data coordinates in the PC space
centers_pc = pca.transform(wine_centers);
score_pc=pca.score(X)
print("PCA score:",score_pc)


#plot pca clustering
fig = plt.figure(6,figsize=(12,12))
plt.title('Clustering of the Wine data after PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.scatter(wine_pc[:,0],wine_pc[:,1],s=50,c=wine_labels)  # data
plt.scatter(centers_pc[:,0],centers_pc[:,1],s=200,c='red',marker='X')  # centroids
plt.show()
plt.title('Clustering of the Wine data after PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
sns.scatterplot(x=wine_pc[:,0], y=wine_pc[:,1], hue="Cluster", 
                data=wine_dropped, palette='Paired', s=20)


#PCA2
X2=wine.drop('quality', axis=1).values
y2=wine['quality'].values
X2 = StandardScaler().fit_transform(X2)#Normalizing datas
pca2 = PCA(n_components=2)
principalComponents = pca2.fit_transform(X2)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDF=pd.concat([principalDf,wine['quality']],axis=1)
score_pc2=pca2.score(X2)
print("PCA2 Score",score_pc2)

#plot pca2 clusters
fig = plt.figure(9,figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
targets = [5,7, 8]#Which original qualitys are the targets
colors = ['r', 'g','b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDF['quality'] == target
    ax.scatter(finalDF.loc[indicesToKeep, 'principal component 1']
               , finalDF.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 10)
ax.legend(targets)
ax.grid()

# Finding optimal cluster number
Max_K = 31  # maximum cluster number
SSE = np.zeros((Max_K-2))  #  array for sum of squares errors
DB = np.zeros((Max_K-2))  # array for Davies Bouldin indeces
for i in range(Max_K-2):
    n_c = i+2
    kmeans = KMeans(n_clusters=n_c, random_state=2020)
    kmeans.fit(X)
    wine_labels = kmeans.labels_
    SSE[i] = kmeans.inertia_
    DB[i] = davies_bouldin_score(X,wine_labels)

#Show optimal cluster number    
fig = plt.figure(10)
plt.title('Sum of squares of error curve')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.plot(np.arange(2,Max_K),SSE, color='red')
plt.show()

# Visualization of DB scores
fig = plt.figure(11)
plt.title('Davies-Bouldin score curve')
plt.xlabel('Number of clusters')
plt.ylabel('DB index')
plt.plot(np.arange(2,Max_K),DB, color='blue')
plt.show()
