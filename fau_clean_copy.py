import os
#import operator
import matplotlib.pyplot as plt
import numpy as np
#import glob
import skimage.io as io
#from skimage import data_dir
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_predict
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
#from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from skimage.io import imread_collection
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#this function reads in images and then flattens them into a dataframe
def flat_images(path):
    images=io.ImageCollection(path)
    images=np.asarray(images)
    print(len(images),"images found")
    flat_images=[]
    for i in range(len(images)):
        img=images[i]
        img1=img.flatten()
        flat_images.append(img1)
    df=pd.DataFrame(flat_images)
    print("Dataframe made")
    return df


    
    

    

#flattens ditylum images
#path=r'1_ditylum/*.tif'
path=r'cnn/1_resized/*.tif'
df_1=flat_images(path)
print("df_1 created")
print(len(df_1))
df_1=df_1.fillna(0)
print("df_1 filled na's with 0's")
df_1['label']=1

print("Label for df_1 changed")

print('Original ditylum images processed')


#flattens non-ditylum images
#path1= r'0_not_ditylum/*.tif'
path1=r'cnn/0_resized/*.tif'
df_0=flat_images(path1)
print("df_0 created")
df_0=df_0.fillna(0)
print("df_0 filled na's with 0's")
df_0['label']=0
print("Label for df_0 changed")


print('Non-Ditylum images processed')

#puts both sets of images in same dataframe
df=df_1.append(df_0,sort=False)
print("Big dataframe made")
df.reset_index(inplace=True,drop=True)
print("index made")
df=df.fillna(0)
print("df filled na's with 0's")

print('Dataframe of all flattened images created')

y=df.label
X=df.drop(['label'],axis=1)
X=scale(X)

print('X has been scaled')

print('X and y are initialized')

# create test and train data (80/20 split)
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=.2,random_state=42)
print("Split complete")


pca=PCA()
pca.fit(X_train)
print("Fit X_train")
csum=np.cumsum(pca.explained_variance_ratio_)
d=np.argmax(csum>=.99)+1
print('Reduces shape to',d)
pca=PCA(n_components=d)
# reduce test set too for metrics later - not in the model itself
X_train_reduced=pca.fit_transform(X_train)
print("Transformed X_train")
X_test_reduced=pca.transform(X_test)
print("Transformed X_test")
print(X_train.shape, X_train_reduced.shape)


def crunch(model,my_x_train,my_y_train,my_x_test,my_y_test,params):
    grid = GridSearchCV(model,params,refit=True,verbose=3,cv=3)    
    grid.fit(my_x_train,my_y_train)
    print(grid.best_params_,grid.best_score_)
    y_pred = grid.predict(my_x_test)
    
    print("Results from " +str(model)+ "in GridSearch:")
    print(accuracy_score(my_y_test,y_pred))
    print(confusion_matrix(my_y_test,y_pred))
    print(classification_report(my_y_test,y_pred))

    return "Model Complete"

models = [LogisticRegression(),KNeighborsClassifier(),SVC()]
lr_params = {'solver':['newton-cg', 'lbfgs'],
        'tol':[1e-2,1e-3,1e-4],
        'random_state':[42],
        'max_iter':[50000],
        'C':np.linspace(0.1,1,10)
        }
knn_params = {'n_neighbors':[116],
        'weights':['uniform','distance'],
        'algorithm':['auto'],
        #'leaf_size':np.array([i for i in range(10,51)]),
        'p':[2]
        }
svm_params = {'kernel':['poly'],
        'random_state':[42],
        'degree':[1,2,3,4,5],
        'gamma':['auto']
        }

my_params = [lr_params,knn_params,svm_params]

for model,param in zip(models,my_params):
    crunch(model,X_train_reduced,y_train,X_test_reduced,y_test,param)

print("EOF")
