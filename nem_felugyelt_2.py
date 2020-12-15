# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 05:06:19 2020

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
X =wine_dropped.values[:, 1:]

#X = wine.drop('quality', axis=1)
x_cols=wine_dropped.columns
#X2=X.to_numpy(copy=True)
y=wine['quality']
#y2=y.to_numpy(copy=True)

f2, ax = plt.subplots(2, 2, figsize=(16, 12))
sns.boxplot('quality', 'alcohol', data=wine, ax=ax[0, 0], palette='Blues')
sns.boxplot('quality', 'sulphates', data=wine, ax=ax[0, 1], palette='Blues')
sns.boxplot('quality', 'volatile acidity', data=wine, ax=ax[1, 0], palette='Blues')
sns.boxplot('quality', 'citric acid', data=wine, ax=ax[1, 1], palette='Blues')

kmeans = KMeans(n_clusters=2, random_state=2020)
kmeans.fit(X)
wine_labels = kmeans.labels_ 
wine_centers = kmeans.cluster_centers_ 
sse = kmeans.inertia_ 
score = kmeans.score(X)
DB = davies_bouldin_score(X,wine_labels)

print(f'Within SSE: {sse}')
print(f'Davies-Bouldin index: {DB}')

# PCA with limited components
pca = PCA(n_components=2)
pca.fit(X)
wine_pc = pca.transform(X)  #  data coordinates in the PC space
centers_pc = pca.transform(wine_centers)  # the cluster centroids in the PC space

# Visualizing of clustering in the principal components space
fig = plt.figure(3);
plt.title('Clustering of the Iris data after PCA');
plt.xlabel('PC1');
plt.ylabel('PC2');
plt.scatter(wine_pc[:,0],wine_pc[:,1],s=50,c=wine_labels);  # data
plt.scatter(centers_pc[:,0],centers_pc[:,1],s=200,c='red',marker='X');  # centroids
plt.legend();
plt.show();

# Kmeans clustering with K=2
kmeans = KMeans(n_clusters=2, random_state=2020);  # instance of KMeans class
kmeans.fit(X);   #  fitting the model to data
wine_labels = kmeans.labels_;  # cluster labels
wine_centers = kmeans.cluster_centers_;  # centroid of clusters
distX = kmeans.transform(X);
dist_center = kmeans.transform(wine_centers);

# Visualizing of clustering in the distance space
fig = plt.figure(4);
plt.title('Iris data in the distance space');
plt.xlabel('Cluster 1');
plt.ylabel('Cluster 2');
plt.scatter(distX[:,0],distX[:,1],s=50,c=wine_labels);  # data
plt.scatter(dist_center[:,0],dist_center[:,1],s=200,c='red',marker='X');  # centroids
plt.legend();
plt.show();

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

# Visualization of SSE values    
fig = plt.figure(5)
plt.title('Sum of squares of error curve')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.plot(np.arange(2,Max_K),SSE, color='red')
plt.show()

# Visualization of DB scores
fig = plt.figure(6)
plt.title('Davies-Bouldin score curve')
plt.xlabel('Number of clusters')
plt.ylabel('DB index')
plt.plot(np.arange(2,Max_K),DB, color='blue')
plt.show()

# The local minimum of Davies Bouldin curve gives the optimal cluster number

# End of code