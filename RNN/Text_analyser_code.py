!pip install gspread





from google.colab import auth

auth.authenticate_user()

import gspread
from oauth2client.client import GoogleCredentials

gc = gspread.authorize(GoogleCredentials.get_application_default())



#data importing ...
worksheet = gc.open('tweets 1').sheet1
# get_all_values gives a list of rows.
rows = worksheet.get_all_values()
x=[]
for i in rows:
    lis=[i[1],i[4]]
    x.append(lis)
import pandas as pd
df=pd.DataFrame(x,columns='sentences responses'.split())

# df contains column of sentences and a column of responses
# response 0 corresponds to negative response , 1 corresponds to positive response and 2 corresponds to neutral ones




# here function count_responses counts number of positive , negative and neutral responses and prints them
def count_responses(x):
    count_1 = count_2 = count_0 = 0
    for i in x:
        if i[1] == '0':
            count_0 += 1
        elif i[1] == '1':
            count_1 += 1
        elif i[1] == '2':
            count_2 += 1
    print(count_0, count_1, count_2)


#count_responses(x)
# in our small project , we had 37680 negative , 32534 positive , 176 neutral examples


# just converting type of responses from str to int
for i in df.responses:
    i = int(i)

#print(df.responses)



# creation of one hot vector of responses
# [1,0,0] , [0,1,0] , [0,0,1]

# joining to data frames 'pd' and 'pd.get_dummies'
df = pd.concat([df,pd.get_dummies(df['responses'], prefix='responses')],axis=1)

#df is shuffled so as training model goes properly
df = df.sample(frac = 1)
#print(df)





# import neccessary modules for training purpose
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#used for tokenizer
vocab_size = 20000
oov_tok = "$tok@"

embedding_dim = 60
max_length = 120

trunc_type='post'# cutting
padding_type='post'# intial to zero from post

#this division of training portion showed good results than else
training_portion = .94



#####PREPROCESSING OF DATA######

#defining training size
train_size = int(len(df.sentences) * training_portion)
#one part is validation set  and other part is test set
part = int((len(df.sentences) * (1 - training_portion)) / 2)

# DIVISION OF EXAMPLES INTO TRAINING , VALIDATION AND TEST SET
train_sentences = df.sentences[:train_size]
#train_sentences yet to be preprocessed....
train_labels = pd.get_dummies(df['responses'], prefix='response')[:train_size]

validation_sentences = df.sentences[train_size:train_size + part]  # for validation this is not test data
#validation_sentences yet to be preprocessed....
validation_labels = pd.get_dummies(df['responses'], prefix='response')[train_size:train_size + part]

test_sentences = df.sentences[train_size + part:]  # for validation this is not test data
test_labels = pd.get_dummies(df['responses'], prefix='response')[train_size + part:]

#print(len(train_sentences))

#print(len(validation_sentences))

#print(len(test_sentences))
#print(train_labels)

## TRAINIG SEET SIZE -66166
## VALIDATION SET SIZE -2111
## TEST SET SIZE - 2113







#creating tokenizer (a dictionary of words from entire examples set)
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(df.sentences)  # word dict formation
word_index = tokenizer.word_index

# conversion of train_sentences str type to array on numbers and padding each example array with 0 after finnshing
train_sequences = tokenizer.texts_to_sequences(train_sentences)  # ( list )
train_padded = pad_sequences(train_sequences, padding='post', maxlen=max_length)

# conversion of validation_sentences str type to array on numbers and padding each example array with 0 after finnshing
validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length)

# conversion of test_sentences str type to array on numbers and padding each example array with 0 after finnshing
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, padding=padding_type, maxlen=max_length)

#print(len(validation_sequences))
#print(validation_padded.shape)

#print(len(word_index))

## Tokenizer is constructed of total 89059 words  ##

## Each example is now converted to (,120) dimensioinal array where each element shows index of that word in tokenizer ##

#print(train_sentences[:5], train_labels[:5])
#print(validation_sentences[:5], validation_labels[:5])
#print(test_sentences[:5], test_labels[:5])

#print(train_padded.dtype)






######BUILDING A MODEL AND TRAINING THE MODEL...########


import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

model = tf.keras.Sequential([  tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
                             #tf.keras.layers.Flatten(),
    tf.keras.layers.Bidirectional(LSTM(40, return_sequences = True)),# find relation in word in sentence
    tf.keras.layers.Bidirectional(LSTM(40,return_sequences = True)),
    #tf.keras.layers.Bidirectional(LSTM(30,return_sequences=False)),
    tf.keras.layers.LSTM(30),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64,activation='relu',kernel_regularizer='l1'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#model.summary()





num_epochs = 10
history = model.fit(train_padded, train_labels, epochs=num_epochs,batch_size=512, validation_data=(validation_padded, validation_labels))

# After trainig completed for 10 epochs ,, traning accuracy was observed 90% and validation accuracy 80%


model.evaluate(test_padded,test_labels)

# Test accuracy showed up 78%


# ploting the graph

import matplotlib.pyplot as plt
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")