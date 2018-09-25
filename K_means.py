from sklearn import datasets
import numpy as np
import pandas as pd

iris = datasets.load_iris()
iris_df= pd.DataFrame(data=np.c_[ iris["data"], iris["target"]],columns=iris["feature_names"]+["target"])

# we have to standardize the data to remove the units
from sklearn.preprocessing import StandardScaler

std = StandardScaler()
X = std.fit_transform(iris_df.values[:,:-1])

#Kmeans algorithm
from sklearn.cluster import KMeans
km =KMeans(n_clusters=3)



#clustering of X values
km.labels_
# predict

#train test split
from sklearn.model_selection import train_test_split
X,x_test,y,y_test = train_test_split(X,iris["target"])
km.fit(X)


#prediction of clustering
correct = 0
for i in range(len(x_test)):
    predict_me = np.array(x_test[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = km.predict(predict_me)
    if prediction[0] == y_test[i]:
        correct += 1

print(correct/len(x_test))