from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import os.path
import sys
import random
import operator
import cPickle as pkl

from keras.preprocessing import sequence
from keras.preprocessing.text import one_hot, Tokenizer_custom as Tokenizer
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import TimeDistributedDense, Dense, Dropout, Activation, RepeatVector, AutoEncoder
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, Callback

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
    
    def on_epoch_end(self, epoch, logs={}):
        f = file(fdir+'/losses.pkl', 'wb')
        pkl.dump(history.losses, f, protocol=pkl.HIGHEST_PROTOCOL)
        f.close()
        
class Sample(Callback):
    def on_epoch_end(self, epoch, logs={}):
        X_eval = X_batch[:int(len(X_data)*0.9)]
        a = model.predict_classes(X_eval[:5],batch_size=1, verbose=1)
        for i,sent in enumerate(a):
            print('original: ',)   
            for word in X_eval[i]:
                if word in X_vocab:
                    sys.stdout.write(X_vocab[word]+' ')
            print('\n')
            print('generated: ',)
            for word in sent:
                if word in Y_vocab:
                    sys.stdout.write(Y_vocab[word]+' ')
            print('\n')
         
def vectorize(data,max_words):#make X vectors out of list of sentences
    print("Indexing the data...")
    tokenizer = Tokenizer(nb_words=max_words, filters='\n')
    tokenizer.fit_on_texts(data)
    data = tokenizer.texts_to_sequences(data)
    data = [element or str(max_words-1) for element in data]
    index = tokenizer.word_index
    inv_index = {v: k for k, v in index.items()}
    inv_index[0]=" "
    inv_index[max_words-1]="<unk>"
    return data,inv_index#,y_data

def shuffle_two(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def pad(vecsx,vecsy):
    #pad to max len and flip
    l = len(max(vecsx,key=len))
    #datax = np.fliplr(sequence.pad_sequences(vecsx,maxlen=l))
    datax = sequence.pad_sequences(vecsx,maxlen=l)
    datay = sequence.pad_sequences(vecsy,maxlen=l)
    datax,datay=shuffle_two(datax,datay)
    print ("Chunks have shape "+str(datax.shape)+str(datay.shape))
    return datax,datay

def chunks(lx,ly, n):
    #split data in chunks in order for vectors to fit in mem 
    n = max(1, n)
    outx=[]
    outy=[]
    chunks_X = [lx[i:i + n] for i in range(0, len(lx), n)]   
    chunks_Y = [ly[i:i + n] for i in range(0, len(ly), n)]   
    for i,chunkx in enumerate(chunks_X):
        chunkx,chunky = pad(chunks_X[i],chunks_Y[i])
        outx.append(chunkx)
        outy.append(chunky)
    return outx,outy

def one_hot(data, max_features):
    print ("Generating one hot vectors..")
    max_len = len(data[0])
    y_data = np.zeros((len(data), max_len, max_features))

    for i, sentence in enumerate(data):
        for t, word in enumerate(sentence):
            y_data[i, t, word] = 1
    print(y_data.shape)
    return y_data

#SETTINGS#
np.random.seed(1337) 
max_features_X = 10000
max_features_Y = 10000
batch_size = 128
nb_layers=2
hidden_size=512
embedding_size=128
nb_epoch=100
maxlen=18

fdir='hotels2'
file1='/titles_sorted.txt'
#file2='/titles_sorted.txt'

X_data,X_vocab = vectorize(open(fdir+file1).readlines(), max_features_X)
np.random.shuffle(X_data)
Y_data,Y_vocab = X_data,X_vocab
#exclude test data
X_train = X_data[:int(len(X_data)*0.9)]
X_test = X_data[int(len(X_data)*0.9):]
Y_train = X_train

X_train, Y_train = zip(*sorted(zip(X_train, Y_train),key=lambda pair: len(pair[0])))
#leave out possible long anomalies
X_train = X_train[:int(len(X_train)*0.97)]
Y_train = Y_train[:int(len(Y_train)*0.97)]
batches_X,batches_Y = chunks(X_train,Y_train, 30000)

print('Building model...')
model = Sequential()
model.add(Embedding(max_features_X, embedding_size, mask_zero=True))
for l in range(nb_layers):
    model.add(LSTM(embedding_size, hidden_size, return_sequences=True))
model.add(TimeDistributedDense(hidden_size,max_features_Y))
model.add(Activation('time_distributed_softmax'))

if os.path.exists(fdir+'/weights.hdf5'):
    model.load_weights(fdir+'/weights.hdf5')
    print (model.shape())
rmsprop=RMSprop(lr=0.0002, rho=0.99, epsilon=1e-8, clipnorm=5)    
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

if (mode =='train'):
    #save all checkpoints        
    checkpointer = ModelCheckpoint(filepath=fdir+"/weights.hdf5", verbose=1, save_best_only=False)
    history = LossHistory()
    sample = Sample()
    
    print("Training...")
    
    for e in range(nb_epoch):
        print("epoch %d" % e)
        #for X_batch,Y_batch in zip(batches_X,batches_Y):
        for i, batch in enumerate(batches_X):
            X_batch= batches_X[i]
            Y_batch = one_hot(batches_Y[i],max_features_Y)
            model.fit(X_batch, Y_batch, batch_size=batch_size, nb_epoch=1, validation_split=0.1, callbacks=[checkpointer,history,sample])
            f = file(fdir+'/losses.pkl', 'wb')
            pkl.dump(history.losses, f, protocol=pkl.HIGHEST_PROTOCOL)
            f.close()

else:
    preds = model.predict_classes(X_test,batch_size=1, verbose=1)

    print(preds[0])
    get_activations = theano.function([model.layers[3].input], model.layers[4].output(train=False), allow_input_downcast=True)
    activations = get_activations(X_test)
    
    
    
    print (activations.shape)
    for s in activations:
        for dim in s[0]:
            sys.stdout.write(str(dim)+' ')
        print('\n')
