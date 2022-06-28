# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 10:40:24 2022

@author: Daniel Franco López
"""

from gensim.models import Word2Vec
import json
import re
from string import punctuation
import os
import multiprocessing
import sys



def read_json(json_filepath):
    f = open(json_filepath)
    return json.load(f)




def process_raw_text(text):
    punct_list = '!?'
    ec_list = "".join([e for e in punctuation if not e in punct_list and not e in ".'-_"] )
    
    text = [ e for e in (text.lower().replace("\n",".").replace("\t",".").translate(str.maketrans(punct_list,"."*len(punct_list)))
        .translate(str.maketrans("","",ec_list))).split(".")
            if e!="" and e!=" "]


    sentences = [  [ word for word in sentence.split(" ") if word !=""]  
                 for sentence in text if sentence !=" " and sentence !=""]
    return text, sentences

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
        
    sep_list = "!?."
    text2 = text
    for char in sep_list:
        text2 = text2.replace(char,char+"$")
        
    sentences = [ [ word for word in sentence.split(" ") if word!=""]  
                 for sentence in text2.split("$") if sentence !=" " and sentence !=""]
    return text,sentences


def generate_data_no_punctuation(data,save_filepath):
    gameDict=dict()
    sentences = list() #se usará para generar el modelo word2vec
    
    def normalize_string(string):
        return string.translate(str.maketrans("","",punctuation)).replace(" ","_")
    
    i=0
    j=0
    for gameText in data.values():
        game = normalize_string(gameText["game"][0])
        game_title = normalize_string(gameText["title"])
        clear_text,wordList = process_raw_text(gameText["text"])
        i += sum([len(sentence) for sentence in wordList])
        j +=1
        #Guardamos el texto limpio y separado por frases
        if game in gameDict:
            gameDict[game].append((game_title,clear_text))
        else:
            gameDict[game] = [(game_title,clear_text),]
            
        #Añadimos la lista de palabras a sentences
        sentences = sentences + wordList

    print("Se han procesado {} textos...".format(j))
    print("Se ha generado un conjunto con {} palabras totales".format(i))
    #Guarda los textos limpios frase a frase en el directorio save_filepath/<juego> 
    for key in gameDict:

        path = save_filepath+"/"+key
        if not os.path.exists(path):
            os.makedirs(path)
        for (title,text) in gameDict[key]:
            if len(title) > 50:
                title = title[:49]
            with open(path+"/"+title+".txt","w") as f:
                for line in text:
                    f.write(line+'\n')
            
    return sentences

def generate_data_with_punctuation(data,save_filepath):
    gameDict=dict()
    texts = list() #se usará para generar el modelo word2vec
    
    #Usaremos esta función auxiliar para poder usar las strings como nombres de ficheros/carpetas
    def normalize_string(string):
        return string.translate(str.maketrans("","",punctuation)).replace(" ","_")
    
    i=0
    j=0
    for gameText in data.values():
        game = normalize_string(gameText["game"][0])
        game_title = normalize_string(gameText["title"])
        clear_text,wordList = process_raw_text_wp(gameText["text"])
        i += sum([len(sentence) for sentence in wordList])
        j += 1
        if game in gameDict:
            gameDict[game].append((game_title,clear_text))
        else:
            gameDict[game] = [(game_title,clear_text),]
            
        #Añadimos la lista de palabras a sentences
        texts = texts + wordList
        
    print("Se han procesado {} archivos...".format(j))
    print("Se ha generado un conjunto con {} palabras totales".format(i))
    #Guarda los textos limpios frase a frase en el directorio save_filepath/<juego> 
    for key in gameDict:

        path = save_filepath+"/"+key
        if not os.path.exists(path):
            os.makedirs(path)
        for (title,text) in gameDict[key]:
            if len(title) > 50:
                title = title[:49]
            with open(path+"/"+title+".txt","w") as f:
                f.write(text)
            
    return texts

def create_word2vec_model(modelname,data,embedding_dim=100,window_size=6,epochs=15,min_word_count=5
                          ,num_cores = multiprocessing.cpu_count(), sample = 1e-5, sg=1):
    model = Word2Vec(
        data,
        workers = num_cores,
        vector_size = embedding_dim,
        min_count=min_word_count,
        window=window_size,
        sample= sample,
        sg=sg,
        epochs=epochs
    )
    model.save("{}_dim{}_w{}_e{}".format(*[modelname,embedding_dim,window_size,epochs]))
    return model

def pipeline(raw_data_filepath,data_filepath,wp=True, create_word2vec= False,modelname=None):
    raw_data = read_json(raw_data_filepath)

    if wp:
        dataset = generate_data_with_punctuation(raw_data, data_filepath)
    else:
        dataset = generate_data_no_punctuation(raw_data, data_filepath)

    if create_word2vec:
        word2vec = create_word2vec_model(modelname, dataset,window_size=2)
        return word2vec
    else:
        return dataset
    
def main():
    args = sys.argv[1:]
    raw_data_filepath,data_filepath = args[0],args[1]
    
    wp = args[2].lower() == "true"
    
    if len(args) <4:
        pipeline(raw_data_filepath, data_filepath,wp)
        return 0
    
    else:
        create_word2vec =  args[3].lower() == "true"
        modelname = args[4]
        model = pipeline(raw_data_filepath, data_filepath,wp=wp,create_word2vec=(create_word2vec),modelname=(modelname))
        print(model.wv.most_similar("dragon"))
        return 0
      
#python preprocesado_inicial.py data/raw_TES.json data/with_punctuation True True w2v_wp 

if __name__ == "__main__":
    main()  

     
