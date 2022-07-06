# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 21:20:31 2022

@author: Daniel Franco López
"""

#IMPORTS
from gensim.models import Word2Vec
import gensim.downloader
import numpy as np
import re
import sys
from string import punctuation
import tensorflow as tf
from random import choice
from tkinter import *

tf.get_logger().setLevel('ERROR')

#Funciones para la generacion letra a letra

def char2token_layer(vocab):
    return tf.keras.layers.StringLookup(num_oov_indices=0,vocabulary=vocab)

def token2char_layer(vocab):
    return tf.keras.layers.StringLookup(num_oov_indices=0,vocabulary=vocab,   invert=True )

def load_vocab(vocab_filepath):
    vocab=list()
    with open(vocab_filepath,"r") as f:
        for char in f.readlines()[0]:
            vocab.append(char)
    return vocab

#Funciones para la generación palabra a palabra:
    
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


#FUNCIONES GENERALES

def sample(preds,temp):
    preds = np.log(preds) / temp
    exp_preds = (np.exp(preds)).astype("float64")
    preds = exp_preds / np.sum(exp_preds)
    probs = np.random.multinomial(1,preds,1)
    return np.argmax(probs)
    
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

#Funciones generadoras de textos

def generate_char(initial_text,vocab,generator,nchars,wp,temp,greedy=False):
    if wp:
        initial_text = process_raw_text_wp(initial_text)
        
    else:
        initial_text = process_raw_text(initial_text)[0]

    char2token = char2token_layer(vocab)
    token2char = token2char_layer(vocab)
    vocab_size = len(vocab)
    tokens = [e for e in initial_text]
    tokens = tf.expand_dims(tf.one_hot(char2token(tokens),depth=vocab_size),0)
    
    for _ in range(nchars):
        preds = generator.predict(tokens)[0]
        if greedy:
            ntoken = np.argmax(preds)
        else:
            ntoken = sample(preds,temp)
        initial_text+=token2char(ntoken)

        ntoken = tf.expand_dims(tf.expand_dims(tf.one_hot(ntoken, depth=vocab_size),0),0)
        tokens = tf.concat([tokens,ntoken],1)
        
        
    return initial_text

def generate_word(initial_text,vocab,generator,n_words,wp,temp,greedy=False):
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

        predicted = generator.predict(tokens, verbose=0)[0]

        if greedy:
            pred_token = np.argmax(predicted)
        else:
            pred_token = sample(predicted,temp)
            
        tokens = np.append(tokens,pred_token).reshape((1,tokens.shape[1]+1))
        
        initial_text += " "+token2word(pred_token)

    return initial_text  



def create_interface():
    
    def call_generate_text():
        v = Toplevel()
        v.title("Texto generado")
        sc = Scrollbar(v)
        sc.pack(side=RIGHT, fill=Y)
        text_panel = Text(v)
        text_panel.insert(INSERT, "Generando texto...")
        text_panel.pack()
        
        mode = value_char_or_word.get()
        generator = tf.keras.models.load_model(value_generator.get())
        wp = 1 == value_wp.get()
        num_units = int(value_num.get())
        temp = float(value_temp.get())
        initial_text = value_text.get()
        if mode: #palabra
            vocab,_ = load_word2vec(value_vocab.get())
            generated_text = generate_word(initial_text, vocab, generator, num_units, wp, temp)
        else:    #caracter
            vocab = load_vocab(value_vocab.get())
            generated_text = generate_char(initial_text, vocab, generator, num_units, wp, temp)
        
        generated_text = generated_text.numpy().decode("utf-8")
        
        text_panel.delete("1.0", END)
        text_panel.insert(INSERT,generated_text)

        
    root = Tk()
    
    root.title("Generador de Textos")
    
   
    label_char_or_word = Label(root,text="Modo de generación: ")
    value_char_or_word = IntVar() #0 caracteres | 1 palabras
    spinbox_char_or_word1 = Radiobutton(root,text="Carácter a carácter",
                                    variable=value_char_or_word,value=0)
    spinbox_char_or_word2 = Radiobutton(root,text="Palabra a palabra",
                                    variable=value_char_or_word,value=1)
    label_char_or_word.grid(row=0,column=0)
    spinbox_char_or_word1.grid(row=0,column=1)
    spinbox_char_or_word2.grid(row=1,column=1)
    
    
    label_vocab = Label(root, text = "Ruta del vocabulario / modelo word2vec: ")
    value_vocab = Entry(root)
    label_vocab.grid(row=2,column=0)
    value_vocab.grid(row=2,column=1)
    
    label_generator = Label(root, text = "Ruta del modelo generador: ")
    value_generator = Entry(root)
    label_generator.grid(row=3,column=0)
    value_generator.grid(row=3,column=1)
    
    label_wp = Label(root, text= "Generación con signos de puntuación: ")
    value_wp = IntVar() #0 np | 1 wp
    checkbutton_wp = Checkbutton(root,text="Activado",variable=value_wp,
                                 onvalue=1,offvalue=0)
    label_wp.grid(row=4,column=0)
    checkbutton_wp.grid(row=4,column=1)
    
    label_num = Label(root, text= "Número de unidades a generar: ")
    value_num = Entry(root)
    label_num.grid(row=5,column=0)
    value_num.grid(row=5,column=1)
    
    label_temp = Label(root, text="Factor temperatura utilizado durante el muestreo: ")
    value_temp = Entry(root)
    label_temp.grid(row=6,column=0)
    value_temp.grid(row=6,column=1)
    
    label_text = Label(root,text="Texto inicial: ")
    value_text = Entry(root)
    label_text.grid(row=7,column=0)
    value_text.grid(row=7,column=1)
    
    submit_btn = Button(root,text="Generar texto",command= call_generate_text)
    submit_btn.grid(row=8,column=1)
    
    root.mainloop()

def main():
    create_interface()
    
    # args = sys.argv[1:]
    # model   = tf.keras.models.load_model(args[2])
    # wp = args[3].lower() == "true"
    # n_units = int(args[4])
    # temp = float(args[5])
    # initial_text = "".join([ word+" " for word in args[6:]])
    # print("Texto inicial: \n",initial_text)
 
    # #word a word
    # if args[0] == "1":

    #     vocab,_ = load_word2vec(args[1])
    #     text = generate_word(initial_text,vocab,model,n_units,wp,temp)
    #     print(text.numpy())

        
    # #char a char
    # elif args[0] == "0":
    #     vocab = load_vocab(args[1])
    #     text = generate_char(initial_text,vocab,model,n_units,wp,temp)
    #     print(text.numpy())

if __name__ == "__main__":
    main()  

#python generar_textos.py 0 models/vocab_all_np.txt models/char_tg_all_wpFalse_n250_l1_rFalse_dFalse_e250 False 250 1 I went to the city
#python generar_textos.py 1 w2v_wp_dim100_w6_e15 models/word_tg_skyrim_w2v6_wpTrue_ws6_n250_l1_rl2_d0.1 True 50 0.8 The dragon of Winterhold