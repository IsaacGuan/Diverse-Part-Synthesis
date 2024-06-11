import tensorflow as tf

def leaky_relu(x, alpha=0.2):
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    x = x - tf.constant(alpha, dtype=tf.float32) * negative_part
    return x

def batch_norm(x, phase_train):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_train)

def fc(inputs, out_dim, phase_train, scope):
    assert len(inputs.get_shape()) == 2
    with tf.name_scope('fc'):
        with tf.name_scope('weights'):
            weights = tf.get_variable(name=scope + '/weights', dtype=tf.float32, shape=[inputs.get_shape()[1], out_dim], initializer=tf.contrib.layers.xavier_initializer(), trainable=phase_train)
            tf.summary.histogram(scope + '/weights', weights)
        with tf.name_scope('biases'):
            biases = tf.get_variable(name=scope + '/biases', shape=[out_dim], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=phase_train)
            tf.summary.histogram(scope + '/biases', biases)
        with tf.name_scope('fc_out'):
            fc = tf.nn.bias_add(tf.matmul(inputs, weights), biases)
    print('fc', 'in', inputs.shape, 'out', fc.shape)
    return fc

def conv3d(inputs, out_dim, phase_train, scope, k_h=4, k_w=4, k_d=4, d_h=2, d_w=2, d_d=2, padding='SAME'):
    with tf.name_scope('conv3d'):
        with tf.name_scope('weights'):
            weights = tf.get_variable(name=scope + '/weights', shape=[k_d, k_h, k_w, inputs.get_shape()[-1], out_dim], initializer=tf.contrib.layers.xavier_initializer(), trainable=phase_train)
            tf.summary.histogram(scope + '/weights', weights)
        with tf.name_scope('conv3d_out'):
            conv = tf.nn.conv3d(inputs, weights, strides=[1, d_d, d_h, d_w, 1], padding=padding)
        print('conv3d', 'in', inputs.shape, 'out', conv.shape)
    return conv

def deconv3d(inputs, out_dim, output_shape, phase_train, scope, k_h=4, k_w=4, k_d=4, d_h=2, d_w=2, d_d=2, padding='SAME'):
    with tf.name_scope('deconv3d'):
        with tf.name_scope('weights'):
            weights = tf.get_variable(name=scope + '/weights', shape=[k_d, k_h, k_w, out_dim, inputs.get_shape()[-1]], initializer=tf.contrib.layers.xavier_initializer(), trainable=phase_train)
            tf.summary.histogram(scope + '/weights', weights)
        with tf.name_scope('deconv3d_out'):
            conv_trans = tf.nn.conv3d_transpose(inputs, weights, output_shape, strides=[1, d_d, d_h, d_w, 1], padding=padding)
        print('deconv3d', 'in', inputs.shape, 'out', conv_trans.shape)
    return conv_trans
