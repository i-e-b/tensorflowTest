import numpy as np
import tensorflow as tf

# Declare a last of features. We only have one real-valued feature.
# There are many other types of columns that are more complicated an useful.
features = [tf.contrib.layers.real_valued_column("", dimension=1)]

# An estimator is the front end to invoke training (fitting) and evaluation (inference).
# There are many predefined type like linear regression, logistic regression,
# linear classification, logistic classification, and many neural network classifiers and regressors.
# The following code provides an estimator that deos linear regression:
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

# TensorFlow provides many helper methods to read and set up data sets.
# Here we use a simple manual Dataset. The various readers provided by
# tf.learn are better at batching and feeding the data in a computationally
# efficient way compared to this simple approach.
dataSet = tf.contrib.learn.datasets.base.Dataset(
    data=np.array([[1],[2],[3],[4]]),
    target=np.array([[0],[-1],[-2],[-3]]))

# We can invoke 1000 training steps by invoking the `fit` method and passing the
# training data set
estimator.fit(x=dataSet.data, y=dataSet.target, steps=1000)

# Here we evaluate how well our model did. In a real example, we would want
# to use a separate validation and testing data set to avoid overfitting.
print(estimator.evaluate(x=dataSet.data, y=dataSet.target))

# Note -- not sure how to get the parameters of the estimator back out.
