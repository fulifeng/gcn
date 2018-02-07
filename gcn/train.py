from __future__ import division
from __future__ import print_function

import copy
import math
from sklearn import metrics
import time
import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN, MLP

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 1e-1, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 20, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

features = np.genfromtxt(
    '/home/ffl/nus/MM/po_hypergraph/data/cur/feature.csv',
    # '/home/ffl/nus/MM/po_hypergraph/data/mvp/jingyuan/feature.csv',
    dtype=float, delimiter=','
)
# calculate adjacent matrix
euc_dis = metrics.pairwise.euclidean_distances(features, features)
median = float(np.median(euc_dis))
adj = infinity_matrix = metrics.pairwise.rbf_kernel(features,
                                                    gamma= 0.5 / (median ** 2))
adj -= np.identity(features.shape[0])
print('adj finished')

ground_truth = np.genfromtxt(
    '/home/ffl/nus/MM/po_hypergraph/data/cur/ground_truth.csv',
    # '/home/ffl/nus/MM/po_hypergraph/data/mvp/jingyuan/ground_truth.csv',
    delimiter=',', dtype=float
)

# y_train = np.genfromtxt(
# '/home/ffl/nus/MM/po_hypergraph/data/cur/ground_truth.csv',
# # '/home/ffl/nus/MM/po_hypergraph/data/mvp/jingyuan/ground_truth_1.csv',
#     delimiter=',', dtype=float
# )[np.newaxis].T
y_train_array = np.genfromtxt(
    '/home/ffl/nus/MM/po_hypergraph/data/cur/ground_truth_1.csv',
    # '/home/ffl/nus/MM/po_hypergraph/data/mvp/jingyuan/ground_truth_1.csv',
    delimiter=',', dtype=float
)
y_train = np.zeros((features.shape[0], 1), dtype=float)
for i in xrange(features.shape[0]):
    y_train[i][0] = y_train_array[i]
y_val = np.zeros(y_train.shape, dtype=float)

train_mask = np.zeros(features.shape[0])
val_mask = np.zeros(features.shape[0])
for i in xrange(features.shape[0]):
    if abs(y_train[i][0]) < 1e-10:
        y_val[i][0] = ground_truth[i]
        val_mask[i] = 1.0
    else:
        train_mask[i] = 1.0

train_mask = np.array(train_mask, dtype=np.bool)
val_mask = np.array(val_mask, dtype=np.bool)

# testing set
y_test = copy.copy(y_val)
test_mask = np.array(val_mask, dtype=np.bool)

# Some preprocessing
# features = preprocess_features(features)
# print('feature 1 shape:', features[1].shape)
if FLAGS.model == 'gcn':
    # support = [preprocess_adj(adj)]
    support = [adj]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

print(adj.shape)
print(features.shape)
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)
print(train_mask.shape, type(train_mask))
print(val_mask.shape, type(val_mask))
print(test_mask.shape, type(test_mask))

# Define placeholders
placeholders = {
    # 'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    # 'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'support': [tf.placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.placeholder(tf.float32, shape=(None, features.shape[1])),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=())
    # 'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}
# print(len(placeholders['support']))

# Create model
# model = model_func(placeholders, input_dim=features[2][1], logging=True)
model = model_func(placeholders, input_dim=features.shape[1], logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.outputs], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    # feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict = construct_feed_dict(features, support, y_train, train_mask,
                                    placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs],
                    feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
