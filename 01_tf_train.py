import tensorflow as tf

# Model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# The model to train and its loss estimator
linear_model = W * x + b
loss = tf.reduce_sum(tf.square(linear_model - y))

# Optimising/training/learning function
optimiser = tf.train.GradientDescentOptimizer(0.025)
train = optimiser.minimize(loss)

# Training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # set our wrong starting values


# Show the pre-training values
in_W, in_b, in_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(in_W, in_b, in_loss))

# Training loop (run the graph multiple times to improve the 'W' and 'b' parameters to reduce the 'loss' value)
for i in range(500):
    sess.run(train, {x:x_train, y:y_train})


# Now print out the result of the training
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
