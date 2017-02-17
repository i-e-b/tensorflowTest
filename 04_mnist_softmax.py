# https://www.tensorflow.org/get_started/mnist/beginners

from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))   # Weights
b = tf.Variable(tf.zeros([10]))        # Biases
y = tf.nn.softmax(tf.matmul(x, W) + b) # the learning model

y_ = tf.placeholder(tf.float32, [None, 10])  # base truth
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# read-out
print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))

# Very basic visualisation of the learned weights
# http://pillow.readthedocs.io/en/4.0.x/
image_node = 127 + (W * 100) # calculation node that gets our image into a 0..255 range centred on 127
img_size = (280,28) # MNIST image sizes
img = Image.frombuffer('F', img_size, sess.run(image_node)).convert('RGB')
img.save('./my.png')
#img.show()
