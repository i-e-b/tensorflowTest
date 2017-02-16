import numpy as np
import tensorflow as tf

# Declare list of features, we only have one real-valued feature
features = [tf.contrib.layers.real_valued_column("", dimension=1)]

# Here is the learning model we will pass to the estimator
# It will be a simple linear model for this example
def model(features, labels, mode, params):
    with tf.device("/cpu:0"):
        # Build a linear model and predict values
        W = tf.get_variable("W", [1], dtype=tf.float64)
        b = tf.get_variable("b", [1], dtype=tf.float64)
        y = W * features[:,0] + b
        # loss sub-graph
        loss = tf.reduce_sum(tf.square(y - labels))
        # Training sub-graph
        global_step = tf.train.get_global_step()
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train = tf.group(optimizer.minimize(loss),
                         tf.assign_add(global_step, 1))
        # ModelFnOps connects subgraphs we built to the appropriate functionality
        return tf.contrib.learn.ModelFnOps(mode=mode, predictions=y, loss=loss, train_op=train) # This is different from the getting started guide

estimator = tf.contrib.learn.Estimator(model_fn=model)

# Define our data set
dataSet = tf.contrib.learn.datasets.base.Dataset(
        data=np.array([[1.],[2.],[3.],[4.]]),
        target=np.array([[0.],[-1.],[-2.],[-3.]]))

# train
estimator.fit(x=dataSet.data, y=dataSet.target, steps=1000)
# evaluate our model
print(estimator.evaluate(x=dataSet.data, y=dataSet.target, steps=10))
