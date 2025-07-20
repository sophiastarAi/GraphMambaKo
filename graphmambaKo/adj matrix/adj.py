import numpy as np
import pandas as pd

edge_attr=np.load("adj matrix\edge_attr.npy")
edge_index=np.load("adj matrix\edge_index.npy")
print(edge_attr.shape)
print(edge_index.shape)

x_data=pd.read_csv("adj matrix/x121.txt",header=None)
y_data=pd.read_csv('adj matrix/y121.txt',header=None)
new_data=pd.concat([x_data,y_data],axis=1)
new_data.columns=['x','y']
adj_mat=np.load('adj matrix/adj_distance.npy')
print(adj_mat)
adj_mat[adj_mat <= 200] = 1
adj_mat[adj_mat > 200] = 0

print(adj_mat)
df=pd.DataFrame(columns=['startX','startY','endX','endY'])
n=adj_mat.shape[0]
for i in range(n):
    for j in range(i+1):
        if adj_mat[i][j]==1:
            startX=new_data.loc[i]['x']
            startY=new_data.loc[i]['y']
            endX=new_data.loc[j]['x']
            endY=new_data.loc[j]['y']
            df=df._append({'startX':startX,'startY':startY,'endX':endX,'endY':endY},ignore_index=True)

df.to_csv('edge_file.csv')