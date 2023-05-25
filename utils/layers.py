import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys
conv1d = tf.layers.conv1d


def Krylov_GCN(inputs, output_dim, norm_mat, activation, in_drop=0.0):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            inputs = tf.nn.dropout(inputs, 1.0 - in_drop)

        inputs = tf.expand_dims(inputs, axis=0)
        seq_fts = tf.layers.conv1d(inputs, output_dim, 1, use_bias=False)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        hops = 20
        seq_fts = tf.squeeze(seq_fts, axis=0)
        n, m = seq_fts.get_shape()
        n = int(n)
        
        H1 = tf.sparse_tensor_dense_matmul(norm_mat, seq_fts)
        subspace = list()
        subspace.append(H1)

        for i in range(hops-1):
            Hi = tf.sparse_tensor_dense_matmul(norm_mat, subspace[i])
            subspace.append(Hi)

        #aggregation
        H = tf.stack(subspace, axis=0)
        H = tf.reshape(H, [hops, n*output_dim])
        H = tf.transpose(H)       
        H = tf.expand_dims(H, axis=0)
        H = tf.layers.conv1d(H, 1, 1, use_bias=False, name="conv")
        H = tf.transpose(H)
        vals = tf.reshape(H, [1, n, output_dim])
        
        #ret = tf.contrib.layers.bias_add(vals)
        bias = tf.get_variable(shape=[output_dim,],initializer=tf.initializers.zeros,name="bias0")
        ret = tf.nn.bias_add(vals,bias)

        return activation(ret)  # activation


def RNNETS_MLP(inputs, output_dim, norm_mat, activation, in_drop=0.0):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            inputs = tf.nn.dropout(inputs, 1.0 - in_drop)

        inputs = tf.expand_dims(inputs, axis=0)
        seq_fts = tf.layers.conv1d(inputs, output_dim, 1, use_bias=False)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        hops = 5
        seq_fts = tf.squeeze(seq_fts, axis=0)
        n, m = seq_fts.get_shape()
        n = int(n)

        H1 = tf.sparse_tensor_dense_matmul(norm_mat, seq_fts)
        subspace = list()
        subspace.append(H1)

        for i in range(hops-1):
            Hi = tf.sparse_tensor_dense_matmul(norm_mat, subspace[i])
            subspace.append(Hi)

        #aggregation
        H = tf.stack(subspace, axis=0)
        H = tf.reshape(H, [hops, n*output_dim])
        H = tf.transpose(H)
        H = tf.expand_dims(H, axis=0)
        H = tf.layers.conv1d(H, 32, 1, use_bias=False, activation="relu")
        #H = tf.layers.conv1d(H, 16, 1, use_bias=False, activation="relu")
        H = tf.layers.conv1d(H, 1, 1, use_bias=False)
        H = tf.transpose(H)
        vals = tf.reshape(H, [1, n, output_dim])

        #ret = tf.contrib.layers.bias_add(vals)

        bias = tf.get_variable(shape=[output_dim,],initializer=tf.initializers.zeros,name="bias")
        ret = tf.nn.bias_add(vals,bias)

        return activation(ret)  # activation


def RNNETS_GRU(inputs, output_dim, norm_mat, activation, in_drop=0.0):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            inputs = tf.nn.dropout(inputs, 1.0 - in_drop)

        inputs = tf.expand_dims(inputs, axis=0)
        seq_fts = tf.layers.conv1d(inputs, output_dim, 1, use_bias=False)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        hops = 5
        seq_fts = tf.squeeze(seq_fts, axis=0)
        n, m = seq_fts.get_shape()
        n = int(n)

        H1 = tf.sparse_tensor_dense_matmul(norm_mat, seq_fts)
        subspace = list()
        subspace.append(H1)

        for i in range(hops-1):
            Hi = tf.sparse_tensor_dense_matmul(norm_mat, subspace[i])
            subspace.append(Hi)

        #aggregation
        H = tf.stack(subspace, axis=0)
        H = tf.reshape(H, [hops, n*output_dim])
        H = tf.transpose(H)
        H = tf.reshape(H, [n, output_dim, hops])
        rnn_hidden_size = hops
        cell = tf.nn.rnn_cell.GRUCell(num_units=rnn_hidden_size)
        outputs, last_states = tf.nn.dynamic_rnn(cell, H, dtype=tf.float32)        
        vals = tf.reduce_max(outputs, axis=-1)
        vals = tf.expand_dims(vals, axis=0)

        #ret = tf.contrib.layers.bias_add(vals)
        bias = tf.get_variable(shape=[output_dim,],initializer=tf.initializers.zeros,name="bias1")
        ret = tf.nn.bias_add(vals,bias)

        return activation(ret)  # activation


def linear_layer(inputs, output_dim, activation, in_drop=0.0):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            inputs = tf.nn.dropout(inputs, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(inputs, output_dim, 1, use_bias=False)

        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = seq_fts
        #ret = tf.contrib.layers.bias_add(vals)
        bias = tf.get_variable(shape=[output_dim,],initializer=tf.initializers.zeros,name="bias2")
        ret = tf.nn.bias_add(vals,bias)

        return activation(ret)  # activation

