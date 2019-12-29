
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import GRU
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import numpy as np
import random
import sys
import os

dataset = open("Shukumar.txt",'r',encoding = 'utf-8').read()
chars = sorted(dataset)

total_chars = len(dataset)
vocabulary = len(chars)

print("Total Characters: ", total_chars)
print("Vocabulary: ", vocabulary)

char_to_int = { c:i for i, c in enumerate(chars)}
int_to_char = { i:c for i, c in enumerate(chars)}

# cut the text in semi-redundant sequences of maxlen characters
dataX = []   
dataY = []
seq_len=40
for i in range(0,total_chars-seq_len,skip):   
    dataX.append(dataset[i:i+seq_len])  
    dataY.append(dataset[i+seq_len])    

    total_patterns = len(dataX) 
    print("\nTotal Patterns: ", total_patterns)

    X = np.zeros((total_patterns, seq_len, vocabulary), dtype=np.bool)
    Y = np.zeros((total_patterns, vocabulary), dtype=np.bool)

for pattern in range(total_patterns):
    for seq_pos in range(seq_len):
        vocab_index = char_to_int[dataX[pattern][seq_pos]]
        X[pattern,seq_pos,vocab_index] = 1
        vocab_index = char_to_int[dataY[pattern]]
        Y[pattern,vocab_index] = 1


# build the model: 
#print('Building model...')
model.add(GRU(neurons[0],input_shape=(seq_len, vocabulary)))
model.add(GRU(neurons[0],input_shape=(seq_len, vocabulary), return_sequences=True))
model.add(Dropout(dropout_rate))


for i in xrange(1,hidden_layers):
    if i == (hidden_layers-1):
     model.add(GRU(neurons[i]))
    else:
     model.add(GRU(neurons[i],return_sequences=True))
     model.add(Dropout(dropout_rate))
    
model.add(Dense(vocabulary))
model.add(Activation('softmax'))

RMSprop_optimizer = RMSprop(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=RMSprop_optimizer)
    
    # save model information
model.save('GRUModelX.h5')
f = open('GRUModelInfoX','w+')
f.write(str(seq_len)+" "+str(batch)+" "+str(skip))
f.close()


    
model.summary()
filepath="BestGRUWeightsX.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')



def sample(seed):
    # helper function to sample an index from a probability array
   for i in range(sample_len):
    
            x = np.zeros((1,seq_len, vocabulary))
            for seq_pos in range(seq_len):
                vocab_index = char_to_int[seed[seq_pos]]
                x[0,seq_pos,vocab_index] = 1
            prediction = model.predict(x,verbose=0)

            
            prediction = np.asarray(prediction).astype('float64')
            prediction = np.log(prediction) / temperature   
            
            exp_preds = np.exp(prediction)
            prediction = exp_preds / np.sum(exp_preds)
            
           
            
            RNG_int = np.random.choice(range(vocabulary), p=prediction.ravel())          
            
            next_char = int_to_char[RNG_int] 
            sys.stdout.write(next_char)
            sys.stdout.flush()
            seed = seed[1:] + next_char
            
            print()