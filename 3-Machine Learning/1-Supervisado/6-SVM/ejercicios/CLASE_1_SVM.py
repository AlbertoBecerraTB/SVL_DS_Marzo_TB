# %% [markdown]
# # Support Vector Machines - Exercise 1

# %% [markdown]
# In this exercise, we'll be using support vector machines (SVMs) to build a spam classifier.  We'll start with SVMs on some simple 2D data sets to see how they work.  Then we'll do some pre-processing work on a set of raw emails and build a classifier on the processed emails using a SVM to determine if they are spam or not.

# %% [markdown]
# The first thing we're going to do is look at a simple 2-dimensional data set and see how a linear SVM works on the data set for varying values of C (similar to the regularization term in linear/logistic regression).  Let's load the data.
# ## Exercise 1
# #### 1. Load libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyparsing import lineStart
import seaborn as sb
from scipy.io import loadmat
import seaborn as sns
# %matplotlib inline
import os
os.chdir('3-Machine Learning/1-Supervisado/6-SVM/ejercicios/')

# %% [markdown]
# #### 2. Load data
# Load the file *ejer_1_data1.mat*. Find the way for loading this kind of file.

# %%
raw_data = loadmat('data/ejer_1_data1.mat')
raw_data

# %% [markdown]
# #### 3. Create a DataFrame with the features and target

# %%
data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])
data['y'] = raw_data['y']
data.head()

# %% [markdown]
# #### 4. Plot a scatterplot with the data

# %%
sns.scatterplot(data=data,
                x="X1",
                y="X2",
                hue="y")

plt.show()

# %% [markdown]
# Notice that there is one outlier positive example that sits apart from the others.  The classes are still linearly separable but it's a very tight fit.  We're going to train a linear support vector machine to learn the class boundary.
# 
# #### 5. LinearSVC
# Declare a Linear SVC with the hyperparamenters:
# 
# ```Python
# LinearSVC(C=1, loss='hinge', max_iter=10000)
# ```

# %%
from sklearn import svm
svc = svm.LinearSVC(C=1, loss='hinge', max_iter=10000)
svc

# %% [markdown]
# #### 6. Try the performance (score)
# For the first experiment we'll use C=1 and see how it performs.

# %%
X = data[['X1', 'X2']]
y = data['y'].values.reshape(-1,)

# %%
svc.fit(X, y)
svc.score(X, y)

# %%
y_hat = svc.predict(X)

mask_out = X['X1'].values < 0.5

y_hat_out = y_hat[mask_out]
y_out = y[mask_out]

print("Nosotros hemos predicho {} y el valor real es {}".format(y_hat_out, y_out))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_true=y, y_pred=y_hat)

svc.coef_
svc.intercept_

svc_line = lambda x: - 1/(svc.coef_[0, 1]) * (svc.coef_[0, 0] * x + svc.intercept_[0])
svc_line_imp = lambda x, y: svc.coef_[0, 0] * x + svc2.coef_[0, 1] * y + svc.intercept_[0]

assert (svc_line(0) * svc.coef_[0, 1] * (-1) == svc.intercept_[0]), 'Mal calculo'

sns.scatterplot(data=data,
                x="X1",
                y="X2",
                hue="y")

xx = np.linspace(0, 5)
sns.lineplot(x=xx,
                y=svc_line(xx),
                color='red',
                linestyle='dashed')

plt.show()

# svc.decision_function(X) == svc_line_imp(X['X1'], X['X2'])
# %% [markdown]
# It appears that it mis-classified the outlier.
# 
# #### 7. Increase the value of C until you get a perfect classifier

# %%
# C decreases the regularization. If C is los, more flexible is the model and better generalization
svc2 = svm.LinearSVC(C=500, loss='hinge', max_iter=10000)
svc2.fit(X, y)
svc2.score(X, y)

# %% [markdown]
# This time we got a perfect classification of the training data, however by increasing the value of C we've created a decision boundary that is no longer a natural fit for the data.  We can visualize this by looking at the confidence level for each class prediction, which is a function of the point's distance from the hyperplane.
# 
# #### 8. Plot Decission Function
# Get the `decision_function()` output for the first model. Plot a scatterplot with X1, X2 and a range of colors based on `decision_function()`

# %%
conf_level = svc.decision_function(X)
svc_line2 = lambda x: - 1/(svc2.coef_[0, 1]) * (svc2.coef_[0, 0] * x + svc2.intercept_[0])

plt.scatter(data.X1,
            data.X2,
            alpha = .8,
            c = conf_level,
            cmap = 'seismic')
cbar = plt.colorbar()

xx = np.linspace(-1, 6)
sns.lineplot(x=xx,
                y=svc_line2(xx),
                color='red',
                linestyle='dashed')

plt.show()

svc2.coef_[0, 0] * X[mask_out].values[0, 0] + svc2.coef_[0, 1] * X[mask_out].values[0, 1] + svc2.intercept_

svc2.coef_[0, 0] * X[mask_out].values[0, 0] + svc2.coef_[0, 1] * X[mask_out].values[0, 0] + svc2.intercept_
# %% [markdown]
# #### 9. Do the same with the second model

# %%
conf_level = svc2.decision_function(data[['X1', 'X2']])

plt.scatter(data.X1,
            data.X2,
            alpha = .8,
            c = conf_level,
            cmap = 'seismic')

cbar = plt.colorbar()

plt.show()

# The outlier is classified correctly, but not so confidence about it.

# %% [markdown]
# Now we're going to move from a linear SVM to one that's capable of non-linear classification using kernels.  We're first tasked with implementing a gaussian kernel function.  Although scikit-learn has a gaussian kernel built in, for transparency we'll implement one from scratch.
# 
# ## Exercise 2

# %% [markdown]
# That result matches the expected value from the exercise.  Next we're going to examine another data set, this time with a non-linear decision boundary.

# %% [markdown]
# #### 1. Load the data `ejer_1_data2.mat`

# %%
raw_data = loadmat('data/ejer_1_data2.mat')

# %% [markdown]
# #### 2. Create a DataFrame with the features and target

# %%
data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])
data['y'] = raw_data['y']
data.head()

# %% [markdown]
# #### 3. Plot a scatterplot with the data

# %%
sns.scatterplot(data=data,
                x="X1",
                y="X2",
                hue="y");

# %% [markdown]
# For this data set we'll build a support vector machine classifier using the built-in RBF kernel and examine its accuracy on the training data.  To visualize the decision boundary, this time we'll shade the points based on the predicted probability that the instance has a negative class label.  We'll see from the result that it gets most of them right.

# %% [markdown]
# #### 4. Declare a SVC with this hyperparameters
# ```Python
# SVC(C=100, gamma=10, probability=True)
# ```
# 

# %%
svc = svm.SVC(C=100, gamma=10, probability=True)
svc

# %% [markdown]
# #### 5. Fit the classifier and get the score

# %%
svc.fit(data[['X1', 'X2']], data['y'])
svc.score(data[['X1', 'X2']], data['y'])

# %% [markdown]
# #### 6. Plot the scatter plot and probability of predicting 0 with a [sequential color](https://matplotlib.org/3.1.1/tutorials/colors/colormaps.html)

# %%
# Probability os being 0, green
probaility = svc.predict_proba(data[['X1', 'X2']])[:,0]

fig, ax = plt.subplots(figsize=(12,8))
plt.scatter(data.X1,
            data.X2,
            alpha = .8,
            c = probaility,
            cmap = 'winter')

cbar = plt.colorbar()

# %% [markdown]
# ## Exercise 3

# %% [markdown]
# For the third data set we're given both training and validation sets and tasked with finding optimal hyper-parameters for an SVM model based on validation set performance.  Although we could use scikit-learn's built-in grid search to do this quite easily, in the spirit of following the exercise directions we'll implement a simple grid search from scratch.
# 
# #### 1. Load the data `ejer_1_data3.mat`

# %%
raw_data = loadmat('data/ejer_1_data3.mat')

# %% [markdown]
# #### 2. Create a DataFrame with the features and target

# %%
data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])
data['y'] = raw_data['y']
data.head()

# %% [markdown]
# #### 3. Get the validation data from the dataset `Xval` and `yval`

# %%
data_val = pd.DataFrame(raw_data['Xval'], columns=['X1', 'X2'])
data_val['y'] = raw_data['yval']
data_val.head()

# %% [markdown]
# #### 4. Try different hyperparameters
# You are going to find the best hyperparameters that best fit your model.
# 1. Try C from 0.01 to 100
# 2. Try gamma from 0.01 to 100
# 
# Train the model and then get the score with the validation data. Which combination of hyperparameters trains the best model in validation score?

# %%
C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

best_score = 0
best_params = {'C': None, 'gamma': None}

for C in C_values:
    for gamma in gamma_values:
        svc = svm.SVC(C=C, gamma=gamma)
        svc.fit(data[['X1', 'X2']], data['y'])
        score = svc.score(data_val[['X1', 'X2']], data_val['y'])
        
        if score > best_score:
            best_score = score
            best_params['C'] = C
            best_params['gamma'] = gamma

best_score, best_params

# %% [markdown]
# ## Exercise 4
# Now we'll move on to the second part of the exercise. In this part our objective is to use SVMs to build a spam filter.  In the exercise text, there's a task involving some text pre-processing to get our data in a format suitable for an SVM to handle.  However, the task is pretty trivial (mapping words to an ID from a dictionary that's provided for the exercise) and the rest of the pre-processing steps such as HTML removal, stemming, normalization etc. are already done.  Rather than reproduce these pre-processing steps, I'm going to skip ahead to the machine learning task which involves building a classifier from pre-processed train and test data sets consisting of spam and non-spam emails transformed to word occurance vectors.
# 
# #### 1. Load the data `spamTrain.mat` and `spamTest.mat`

# %%
spam_train = loadmat('data/spamTrain.mat')
spam_test = loadmat('data/spamTest.mat')

spam_test

# %% [markdown]
# #### 2. Create a DataFrame with the features and target, for train and test
# Be careful with the test dimensions

# %%
X_train = spam_train['X']
y_train = spam_train['y'].ravel()

# %%
X_test = spam_test['Xtest']
y_test = spam_test['ytest'].ravel()

# %%
train_features.shape, train_target.shape, train_features.shape, train_target.shape

# %% [markdown]
# #### 3. Fit a SVC and get the accuracy in train and test

# %%
svc = svm.SVC()
svc.fit(X_train, y_train)
print('Training accuracy = {0}%'.format(np.round(svc.score(X_train, y_train) * 100, 2)))
print('Test accuracy = {0}%'.format(np.round(svc.score(X_test, y_test) * 100, 2)))

# %% [markdown]
# Each document has been converted to a vector with 1,899 dimensions corresponding to the 1,899 words in the vocabulary.  The values are binary, indicating the presence or absence of the word in the document.  At this point, training and evaluation are just a matter of fitting the testing the classifer. 


