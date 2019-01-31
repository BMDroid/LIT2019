#!/usr/bin/env python
# coding: utf-8

# In[43]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[44]:


import numpy as np
# import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


# In[45]:


iris = pd.read_csv('dataOriginal.csv')


# In[46]:


iris.shape


# In[47]:


iris.head()


# In[48]:


iris = iris.sort_values(by=['WinOrLose'])


# In[49]:


iris.head()


# In[50]:


plt.scatter(iris[:28].MarksSimilarity, iris[:28].AuralSimilarity, label='Lose')
plt.scatter(iris[-10:].MarksSimilarity, iris[-10:].AuralSimilarity, label='Win')
plt.xlabel('Marks Similarity')
plt.ylabel('Visual Similarity')
plt.legend(loc='best')


# In[51]:


X = iris.drop(labels=['WinOrLose'], axis=1).values
Y = iris.WinOrLose.values


# In[52]:


# set seed for numpy and tensorflow
# set for reproducible results
seed = 5
np.random.seed(seed)
tf.set_random_seed(seed)


# In[53]:


# set replace=False, Avoid double sampling
trainIndex = np.random.choice(len(X), round(len(X) * 0.8), replace=False)


# In[54]:


# diff set
testIndex = np.array(list(set(range(len(X))) - set(trainIndex)))
trainX = X[trainIndex]
trainY = Y[trainIndex]
testX = X[testIndex]
testY = Y[testIndex]


# In[55]:


# Define the normalized function
def min_max_normalized(data):
    colMax = np.max(data, axis=0)
    colMin = np.min(data, axis=0)
    return np.divide(data - colMin, colMax - colMin)


# In[56]:


# Normalized processing, must be placed after the data set segmentation, 
# otherwise the test set will be affected by the training set
trainX = min_max_normalized(trainX)
testX = min_max_normalized(testX)


# In[57]:


# Begin building the model framework
# Declare the variables that need to be learned and initialization
# There are 6 features here, A's dimension is (6, 1)
W = tf.Variable(tf.random_normal(shape=[6, 1]), name='W')
b = tf.Variable(tf.random_normal(shape=[1, 1]), name= 'b')
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# In[58]:


# Define placeholders
data = tf.placeholder(dtype=tf.float32, shape=[None, 6])
target = tf.placeholder(dtype=tf.float32, shape=[None, 1])


# In[59]:


# Declare the model you need to learn
model = tf.matmul(data, W) + b


# In[60]:


# Declare loss function
# Use the sigmoid cross-entropy loss function,
# first doing a sigmoid on the model result and then using the cross-entropy loss function
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model, labels=target))


# In[61]:


# Define the learning rate， batch_size etc.
learning_rate = 0.003
batch_size = 40
iter_num = 10000


# In[62]:


# Define the optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)


# In[63]:


# Define the goal
goal = optimizer.minimize(loss)


# In[64]:


# Define the accuracy
# The default threshold is 0.5, rounded off directly
prediction = tf.round(tf.sigmoid(model))
# Bool into float32 type
correct = tf.cast(tf.equal(prediction, target), dtype=tf.float32)
# Average
accuracy = tf.reduce_mean(correct)
# End of the definition of the model framework


# In[65]:


# Start training model
# Define the variable that stores the result
loss_trace = []
train_acc = []
test_acc = []


# In[66]:


# training model
for epoch in range(iter_num):
    # Generate random batch index
    batchIndex = np.random.choice(len(trainX), size=batch_size)
    batchTrainX = trainX[batchIndex]
    batchTrainY = np.matrix(trainY[batchIndex]).T
    sess.run(goal, feed_dict={data: batchTrainX, target: batchTrainY})
    temp_loss = sess.run(loss, feed_dict={data: batchTrainX, target: batchTrainY})
    # convert into a matrix, and the shape of the placeholder to correspond
    temp_train_acc = sess.run(accuracy, feed_dict={data: trainX, target: np.matrix(trainY).T})
    temp_test_acc = sess.run(accuracy, feed_dict={data: testX, target: np.matrix(testY).T})
    # recode the result
    loss_trace.append(temp_loss)
    train_acc.append(temp_train_acc)
    test_acc.append(temp_test_acc)
    # output
    if (epoch + 1) % 300 == 0:
        print('epoch: {:4d} loss: {:5f} train_acc: {:5f} test_acc: {:5f}'.format(epoch + 1, temp_loss,
                                                                          temp_train_acc, temp_test_acc))


# In[67]:


# Visualization of the results
# loss function
plt.plot(loss_trace)
plt.title('Cross Entropy Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


# In[68]:


# accuracy
plt.plot(train_acc, 'b-', label='train accuracy')
plt.plot(test_acc, 'k-', label='test accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Train and Test Accuracy')
plt.legend(loc='best')
plt.show()


# In[40]:


# vars = tf.trainable_variables()
# vars_vals = sess.run(vars)
# for var, val in zip(vars, vars_vals):
#    print("var: {}, value: {}".format(var.name, val))


# In[41]:


sess.run(W)


# In[42]:


sess.run(b)


# In[ ]:




