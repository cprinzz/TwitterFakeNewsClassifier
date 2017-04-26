import tensorflow as tf
import numpy as np
import pandas as pd
from pandas import DataFrame
from collections import Counter
from sklearn import model_selection
import csv
from model import NewsClassifier
import json
from sklearn import metrics

df = pd.read_csv('news_data2.csv',header=0, index_col=None, encoding='utf8')
X_train, X_test, y_train, y_test = model_selection.train_test_split(df['text_clean'].values,df['target'].values,test_size=.33, random_state=42)
print 'X_train: {}\nx_test: {}\ny_train: {}\ny_test:{}\n'.format(len(X_train),len(X_test),len(y_train),len(y_test))

# for i in range(0,len(X_train)):
#     X_train[i] = X_train[i].encode('utf8')

def getVocab(data):
    vocab = Counter()
    text = ''
    for i in range(0, len(X_train)):
        text += ' ' + X_train[i]
    for i in range(0, len(X_test)):
        text += ' ' + X_test[i]
    for word in text.split(' '):
        word_lower = word.lower()
        vocab[word_lower] += 1
    return vocab

def get_word_2_index(vocab):
    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word] = i
    json.dump(word2index,open('vocab_index.json','w'),encoding='utf8')
    return word2index

def get_batch(X,Y,i,batch_size):
    batches = []
    results = []

    texts = X[i*batch_size:i*batch_size+batch_size]
    categories = Y[i*batch_size:i*batch_size+batch_size]

    for text in texts:
        layer = np.zeros(total_words,dtype=float)
        for word in text.split(' '):
        # if word in word2index.keys():
            layer[word2index[word.lower()]] += 1

        batches.append(layer)

    for category in categories:
        y = np.zeros((2),dtype=float)
        if category == 0:
            y[0] = 1.
        elif category == 1:
            y[1] = 1.

        results.append(y)

    return np.array(batches),np.array(results)

def print_weights_biases():
    print 'Weights:'
    for key in weights.keys():
        print '{} : {}'.format(key, sess.run(weights[key]))
    print 'Biases:'
    for key in biases.keys():
        print '{} : {}'.format(key, sess.run(biases[key]))

vocab = getVocab(X_train)
word2index = get_word_2_index(vocab)
print 'Total words: {}'.format(len(vocab))
print 'Index of \'the\': {}'.format(word2index['the'])

#create placeholder tensors
total_words = len(vocab)
n_input = total_words
n_classes = 2
input_tensor = tf.placeholder(tf.float32,[None, n_input], name='input')
output_tensor = tf.placeholder(tf.float32,[None, n_classes], name='output')
print 'Input tensor dim: {}'.format(input_tensor.get_shape())
print 'Output tensor dim: {}'.format(output_tensor.get_shape())

batch_size = 150
n_hidden_1 = 10
n_hidden_2 = 5
display_step = 1

print("Each batch has 150 texts and each matrix has 119930 elements (words):",get_batch(X_train, y_train,1,150)[0].shape)

weights = {
    'h1': tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


learning_rate = .001

#Construct model

prediction = NewsClassifier.multilayer_perceptron(input_tensor, weights, biases)
#define loss
entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_tensor)
loss = tf.reduce_mean(entropy_loss)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

training_epochs = 10
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess,'tmp/model.ckpt')
    print 'Model restored.'

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(len(X_train)/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x,batch_y = get_batch(X_train, y_train,i,batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            c,_ = sess.run([loss,optimizer], feed_dict={input_tensor: batch_x,output_tensor:batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "loss=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    #Test model
    index_prediction = tf.argmax(prediction,1)
    index_correct = tf.argmax(output_tensor,1)
    correct = tf.equal(index_prediction, index_correct)

    # precision = metrics.precision_score(index_prediction,index_correct)
    # recall = metrics.recall_score(index_prediction,index_correct)
    accuracy = tf.reduce_mean(tf.cast(correct,'float'))
    total_test_data = len(X_test)
    batch_x_test, batch_y_test = get_batch(X_test, y_test, 0,len(X_test))
    print ('Accuracy: ',accuracy.eval({input_tensor:batch_x_test, output_tensor:batch_y_test}))

    save_path = saver.save(sess, "tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)

#TODO: evaluate weights to determine keywords
