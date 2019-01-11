import tensorflow as tf
import numpy as np

# Model parameters
W_out = tf.Variable(tf.zeros([4]), dtype=tf.float32)
W_in = tf.Variable(tf.zeros([3, 4]), dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32, shape=[None, 3])
neural_model = tf.sigmoid(tf.matmul(x, W_in)) * W_out
y = tf.placeholder(tf.float32, shape=[None])

# loss
loss = tf.reduce_sum(tf.square(neural_model - y))  # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = np.random.rand(10000, 3)
y_train = x_train[:, 0] + 2 * x_train[:, 1]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)  # reset values to wrong
for i in range(1000):
    print(i)
    sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W_in, W_out, loss], {x: x_train, y: y_train})

print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))
