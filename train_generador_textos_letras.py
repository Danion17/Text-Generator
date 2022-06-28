# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 17:05:57 2022

@author: Daniel Franco López
"""
#IMPORTS
import numpy as np
import re
import sys
from string import punctuation
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout
from tensorflow.keras.optimizers import RMSprop



#Cargamos los ficheros en listas de oraciones repartiendo en función del juego del que procede el texto
def load_data_sentence(filepath,game,max_docs=False):
    data = list()
    for folder in os.listdir(filepath):
        if folder == game or not game:
            textfiles = os.listdir(filepath+"/"+folder)
            if max_docs and len(textfiles) > max_docs :
                textfiles = textfiles[:max_docs]
            for txt in textfiles:
                with open(filepath+"/"+folder+"/"+txt) as f:
                    for line in f.readlines():
                        data.append(line.replace("\n", ""))                    
    return data

def load_data_text(filepath,game,max_docs=False):
    data = list()
    for folder in os.listdir(filepath):
        if folder == game or not game:
            textfiles = os.listdir(filepath+"/"+folder)
            if max_docs and len(textfiles) > max_docs :
                textfiles = textfiles[:max_docs]
            for txt in textfiles:
                with open(filepath+"/"+folder+"/"+txt) as f:
                    data.append( "".join(f.readlines()))                    
    return data

def char2token_layer(vocab):
    return tf.keras.layers.StringLookup(num_oov_indices=0,vocabulary=vocab)

def token2char_layer(vocab):
    return tf.keras.layers.StringLookup(num_oov_indices=0,vocabulary=vocab,   invert=True )

def process_data(data,window_size,step=1,max_data=100000):
    ndata = list()
    vocab = set()
    for unit in data:
        length = len(unit)
        
        for e in unit:
            vocab.add(e)
        
        for i in range(0,length-window_size+1,step):
            ndata.append([ e for e in unit[i:i+window_size]])
            if len(ndata) == max_data:
                break
        if len(ndata) == max_data:
            break
    vocab = list(vocab)
    vocab.sort()
    print(len(ndata))
    return char2token_layer(vocab)(np.array(ndata)),vocab

def create_dataset(data,vocab_size,train_percent = 0.9):
    i = round(len(data)*train_percent)
    X_train,y_train = tf.one_hot(data[:i,:-1],depth=vocab_size),tf.one_hot( data[:i,-1],depth=vocab_size)
    X_test,y_test   = tf.one_hot(data[i:,:-1],depth=vocab_size),tf.one_hot( data[i:,-1],depth=vocab_size)
    return X_train,y_train,X_test,y_test

    
def create_model(modelname,vocab_size,num_units = 150,num_lstm_layers=2,optimizer=RMSprop(learning_rate=10e-3)
                 ,kernel_reg=False,dropout=False,):
    model= Sequential()
    
    if kernel_reg:
        if num_lstm_layers>1:
            model.add(LSTM(num_units, input_shape=(None,vocab_size),kernel_regularizer=kernel_reg,return_sequences=True ))
            if dropout:
                model.add(Dropout(dropout)) 
            for i in range(num_lstm_layers-1,0,-1):
                if i !=1:
                    model.add(LSTM(num_units, input_shape=(None,num_units),kernel_regularizer=kernel_reg,return_sequences=True))
                else:
                    model.add(LSTM(num_units, input_shape=(None,num_units),kernel_regularizer=kernel_reg))
                if dropout:
                    model.add(Dropout(dropout)) 
        else:
            model.add(LSTM(num_units, input_shape=(None,vocab_size),kernel_regularizer=kernel_reg))
            if dropout:
                model.add(Dropout(dropout)) 
        model.add(Dense(vocab_size,activation="softmax"))
        model.compile(loss="categorical_crossentropy",optimizer=optimizer
                               ,metrics=["acc",])
    else:
        if num_lstm_layers>1:
            model.add(LSTM(num_units, input_shape=(None,vocab_size),return_sequences=True ))
            if dropout:
                model.add(Dropout(dropout)) 
            for i in range(num_lstm_layers-1,0,-1):
                if i !=1:
                    model.add(LSTM(num_units, input_shape=(None,num_units),return_sequences=True))
                else:
                    model.add(LSTM(num_units, input_shape=(None,num_units)))
                if dropout:
                    model.add(Dropout(dropout)) 
        else:
            model.add(LSTM(num_units, input_shape=(None,vocab_size) ))
            if dropout:
                model.add(Dropout(dropout)) 
        model.add(Dense(vocab_size,activation="softmax"))
        model.compile(loss="categorical_crossentropy",optimizer=optimizer
                               ,metrics=["acc",])
    modelname +="_n{}_l{}_r{}_d{}".format(*[num_units,num_lstm_layers,kernel_reg,dropout])
    model.summary()
    return model,modelname

def create_callbacks(early_stop = True,es_patience=15,es_monitor= 'loss',restore_best_weights=True,checkpoint=False,checkpoint_filepath="tmp/checkpoints_chars_text_generator"):
    callback_list =list()
    if early_stop:
        callback_list.append(tf.keras.callbacks.EarlyStopping(monitor=es_monitor, patience=es_patience
                                                              ,restore_best_weights=restore_best_weights))
    if checkpoint:
        callback_list.append( tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,monitor="accuracy"))
    return callback_list
     
def process_raw_text(text):
    punct_list = '!?'
    ec_list = "".join([e for e in punctuation if not e in punct_list and not e in ".'-_"] )
    
    text = [ e for e in (text.lower().replace("\n",".").replace("\t",".").translate(str.maketrans(punct_list,"."*len(punct_list)))
        .translate(str.maketrans("","",ec_list))).split(".")
            if e!="" and e!=" "]

    return text

def add_sep(text,char):
    return re.sub("(?<=[a-zA-Z,),(])\{}(\s|$)".format(char)," {} ".format(char), text)

def process_raw_text_wp(text):
    punct_list = '!?.,;:"()*+|'
    ec_list = "".join([e for e in punctuation if not e in punct_list and not e in "'-_"] )
    
    #text a minusculas y eliminando los elementos de ec_list
    text = text.lower().replace("\n"," | ").replace("\t"," | ").replace("("," ( ").translate(str.maketrans(ec_list," "*len(ec_list)))
    #añadimos separacion a los caracteres en punct_list
    for char in punct_list:
        text = add_sep(text,char)
        
    return text

def save_vocab(vocab,vocab_filepath):
    with open(vocab_filepath,"w") as f:
        for char in vocab:
            f.write(char)
def read_vocab(vocab_filepath):
    vocab=list()
    with open(vocab_filepath,"r") as f:
        for char in f.readlines()[0]:
            vocab.append(char)
    return vocab
def sample(preds,temp):
    preds = np.log(preds) / temp
    exp_preds = (np.exp(preds)).astype("float64")
    preds = exp_preds / np.sum(exp_preds)
    probs = np.random.multinomial(1,preds,1)
    return np.argmax(probs)
    

def generate_text(initial_text,vocab,model,nchars,temp,wp,greedy=False):
    if wp:
        initial_text = process_raw_text_wp(initial_text)
    else:
        initial_text = process_raw_text(initial_text)
    
    char2token = char2token_layer(vocab)
    token2char = token2char_layer(vocab)
    vocab_size = len(vocab)
    tokens = [e for e in initial_text]
    tokens = tf.expand_dims(tf.one_hot(char2token(tokens),depth=vocab_size),0)
    
    for _ in range(nchars):
        preds = model.predict(tokens)[0]
        if greedy:
            ntoken = np.argmax(preds)
        else:
            ntoken = sample(preds,temp)
        initial_text+=token2char(ntoken)

        ntoken = tf.expand_dims(tf.expand_dims(tf.one_hot(ntoken, depth=vocab_size),0),0)
        tokens = tf.concat([tokens,ntoken],1)
        
        
    return initial_text
        
        
def pipeline(datapath,modelname,vocabpath,wp,game=None):
    
    print("Preparando conjunto de entrenamiento")
    if wp:
        data =  load_data_sentence(datapath,game)
    else:
        data =  load_data_text(datapath,game)
        
    window_size = 60
    dataset,vocab = process_data(data,window_size)
        
    X_train,y_train,X_test,y_test = create_dataset(dataset,len(vocab))
    
    print(X_train[0])
    modelname +="_wp{}".format(wp)
    model,modelname = create_model(modelname,len(vocab),num_units = 150,num_lstm_layers=1)
    
    callbacks = create_callbacks(early_stop=True,checkpoint=False)
    epochs = 50
    model.fit(X_train,y_train,epochs=epochs,batch_size=128
                      ,shuffle=True,callbacks=callbacks,
                      validation_split=0.1)
    modelname += "_e{}".format(epochs)
    model.save(modelname)
    
    save_vocab(vocab, vocabpath)
    print(model.evaluate(x=X_test, y=y_test))
    print(generate_text("The dragons went after the",vocab,model,100,1,True))
    
def main():
    args = sys.argv[1:]
    
    wp = args[3].lower() == "true"
    if len(args)>4: 
        pipeline(args[0],args[1],args[2],wp,args[4])
    else:
        pipeline(args[0],args[1],args[2],wp)


pipeline("data/with_punct","models/char_tg_skyrim","models/vocab_with_punct.txt",True,"Skyrim")


# if __name__ == "__main__":
#     main()  
