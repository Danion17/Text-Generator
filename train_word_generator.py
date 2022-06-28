# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 14:05:57 2022

@author: Daniel Franco López
"""
#IMPORTS
from gensim.models import Word2Vec
import gensim.downloader
import numpy as np
import re
from string import punctuation
import os
import sys
import tensorflow as tf
from random import choice
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout,Embedding
from tensorflow.keras.optimizers import RMSprop


tf.get_logger().setLevel('ERROR')

def load_word2vec(word2vec_filename):
    
    #Cargamos el modelo word2vec entrenado
    model = Word2Vec.load(word2vec_filename)
    
    embedding_weights = model.wv.vectors   #vectores del word embedding | dimension ()
    vocab = np.array(model.wv.index_to_key)        #array del vocabulario
    

    return vocab,embedding_weights


def word2token_layer(vocab):

    input_layer = tf.keras.layers.StringLookup(num_oov_indices=0,vocabulary=vocab)
    return input_layer

def token2word_layer(vocab):

    output_layer = tf.keras.layers.StringLookup(num_oov_indices=0,vocabulary=vocab, invert=True)
    return output_layer


def load_data(filepath,foldername,vocab,max_docs=False):
    data = list()
    w2t_layer = word2token_layer(vocab)
    for folder in os.listdir(filepath):
        if not foldername or folder == foldername: 
            textfiles = os.listdir(filepath+"/"+folder)
            if max_docs and len(textfiles) > max_docs :
                textfiles = textfiles[:max_docs]
            for txt in textfiles:
                with open(filepath+"/"+folder+"/"+txt) as f:
                    for line in f.readlines():
                        sentence = [word for word in line.replace("\n","").split(" ") if word!=""]
                        cond = sum([1 for word in sentence if word not in vocab])
                        if not cond:
                            data.append(w2t_layer(sentence))
                                       
    return data


def load_data_wp(filepath,foldername,vocab,max_docs=False):
    data = list()
    w2t_layer = word2token_layer(vocab)
    for folder in os.listdir(filepath):
        if not foldername or folder == foldername: 
            textfiles = os.listdir(filepath+"/"+folder)
            if max_docs and len(textfiles) > max_docs :
                textfiles = textfiles[:max_docs]
            for txt in textfiles:
                
                with open(filepath+"/"+folder+"/"+txt) as f:
                    for line in f.readlines():
                        sentence = [word for word in line.split(" ") if word!=""]
                        cond = sum([1 for word in sentence if word not in vocab])
                        if not cond:
                            data.append(w2t_layer(sentence))
                            
    return data

def data_windowing(data,window_size,step=1,max_data=None):
    ndata = list()
    for sentence in data:
        length = len(sentence)
        for i in range(0,length-window_size+1,step):
            ndata.append(sentence[i:i+window_size])
            if len(ndata) == max_data:
                break
        if len(ndata) == max_data:
            break
    return np.array(ndata)

def create_dataset(data,vocab_size,train_percent = 0.9):
    i = round(len(data)*train_percent)
    X_train,y_train = data[:i,:-1],tf.one_hot( data[:i,-1],depth=vocab_size)
    X_test,y_test   = data[i:,:-1],tf.one_hot( data[i:,-1],depth=vocab_size)
    return X_train,y_train,X_test,y_test

    
def create_model(modelname,vocab_size,embedding_dim,embedding_weights,
                          num_units = 150,num_lstm_layers=1,optimizer=RMSprop(learning_rate=10e-3),
                          dropout_rate=False,kernel_reg=False):
    text_generator= Sequential()
    
    embedding_layer = Embedding(vocab_size,embedding_dim,input_shape=(None,),
                                               trainable=False)
    
    text_generator.add(embedding_layer)
    embedding_layer.set_weights([embedding_weights])
    
    
    if kernel_reg:
        if num_lstm_layers>1:
        
            text_generator.add(LSTM(num_units, input_shape=(None,embedding_dim),kernel_regularizer=kernel_reg,return_sequences=True ))
            if dropout_rate:
                text_generator.add(Dropout(dropout_rate))
            for i in range(num_lstm_layers-1,0,-1):
                if i!=1:
                    text_generator.add(LSTM(num_units, input_shape=(None,num_units),kernel_regularizer=kernel_reg,return_sequences=True ))
                else:
                    text_generator.add(LSTM(num_units, input_shape=(None,num_units),kernel_regularizer=kernel_reg))
                if dropout_rate:
                    text_generator.add(Dropout(dropout_rate))
        else:
    
            text_generator.add(LSTM(num_units, input_shape=(None,embedding_dim),kernel_regularizer=kernel_reg ))
            if dropout_rate:
                text_generator.add(Dropout(dropout_rate))
        
    else:
        if num_lstm_layers>1:
        
            text_generator.add(LSTM(num_units, input_shape=(None,embedding_dim),return_sequences=True ))
            if dropout_rate:
                text_generator.add(Dropout(dropout_rate))
            for i in range(num_lstm_layers-1,0,-1):
                if i!=1:
                    text_generator.add(LSTM(num_units, input_shape=(None,num_units),return_sequences=True ))
                else:
                    text_generator.add(LSTM(num_units, input_shape=(None,num_units)))
                if dropout_rate:
                    text_generator.add(Dropout(dropout_rate))
        else:
    
            text_generator.add(LSTM(num_units, input_shape=(None,embedding_dim)))
            if dropout_rate:
                text_generator.add(Dropout(dropout_rate))
        
        
    text_generator.add(Dense(vocab_size,activation="softmax"))
    text_generator.compile(loss="categorical_crossentropy",optimizer=optimizer
                           ,metrics=["acc",])
    modelname +="_n{}_l{}_r{}_d{}".format(*[num_units,num_lstm_layers,kernel_reg,dropout_rate])
    text_generator.summary()
    return text_generator,modelname

def create_callbacks(early_stop = True,es_patience=10,es_monitor= 'loss',restore_best_weights=True,checkpoint=False,checkpoint_filepath="tmp/checkpoints_v3"):
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


def sample(preds,temp):
    preds = np.log(preds) / temp
    exp_preds = (np.exp(preds)).astype("float64")
    preds = exp_preds / np.sum(exp_preds)
    probs = np.random.multinomial(1,preds,size=1)
    return np.argmax(probs)


    
def generate_text(initial_text,vocab,wp,n_words,generator,temp,greedy=False):
    if wp:
        sentence= process_raw_text_wp(initial_text).split(" ")
    else:
        sentence= ("".join(process_raw_text(initial_text))).split(" ")
    print(sentence)
    for i in range(len(sentence)):
        print(sentence[i] )
        if sentence[i] not in vocab:
            sentence[i] = choice(vocab)
    word2token= word2token_layer(vocab)
    token2word= token2word_layer(vocab)
    tokens = np.array([word2token(sentence),])
    for _ in range(n_words):
        # print(tokens)
        predicted = generator.predict(tokens, verbose=0)[0]
        # print(predicted)
        if greedy:
            pred_token = np.argmax(predicted)
        else:
            pred_token = sample(predicted,temp)
        # print(pred_token)
        tokens = np.append(tokens,pred_token).reshape((1,tokens.shape[1]+1))
        
        initial_text += " "+token2word(pred_token)

    return initial_text                    


def pipeline(datapath,w2vpath,modelname,wp,game=False):
    

    vocab,embedding_weights = load_word2vec(w2vpath)
    print("Cargando datos..")
    if wp:
        data = load_data_wp(datapath,game, vocab)
    else:
        data = load_data(datapath,game, vocab)
        
    print("Generando conjuntos de entrenamiento...")
    window_size = 2
    X_train,y_train,X_test,y_test = create_dataset(data_windowing(data, window_size,max_data=25000),len(vocab))
    data=0

    print(X_train.shape)    
    # token2word= token2word_layer(vocab)
    
    # print(data[:5])
    # print(token2word(data[0]))
    
    
    print(X_train.shape)
    modelname += "_wp{}_ws{}".format(wp,window_size)
    text_generator,modelname = create_model(modelname,len(vocab), embedding_weights.shape[1],embedding_weights,
                        num_units=250,num_lstm_layers=1,kernel_reg=False,dropout_rate=(False))
    callbacks = create_callbacks(early_stop=False,checkpoint=False)
    text_generator.fit(X_train,y_train,epochs=250,batch_size=128
                      ,shuffle=True,callbacks=callbacks,
                      validation_split=0.1)
    text_generator.save(modelname)
    print(text_generator.evaluate(x=X_test, y=y_test))
    print(generate_text("|The dragons went after ",vocab,True,15,text_generator,1))
    
    
def main():
    args = sys.argv[1:]


    wp = args[3].lower() == "true"

    if len(args) == 4:
        pipeline(args[0],args[1],args[2],wp)
    elif len(args) == 5:

        pipeline(args[0],args[1],args[2],wp,args[4])
    else:
        raise Exception("Número de argumentos erroneos. Args: \n datapath w2vpath modelname wp game=None")
        

if __name__ == "__main__":
    main()  


#python train_word_generator.py data/no_punct w2v_wp_dim100_w2_e15 models/word_tg_skyrim_w2v2  True Skyrim
