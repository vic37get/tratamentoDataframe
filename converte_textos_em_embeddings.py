import multiprocessing
import os
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer


def get_text_split(text, length=200, overlap=50):
    l_total = []
    l_parcial = []
    n_words = len(text.split()) 
    n = n_words//(length-overlap)+1 
    if n_words % (length-overlap) == 0:
        n = n-1
    if n ==0:
        n = 1
    for w in range(n):
        if w == 0:
            l_parcial = text.split()[:length]
        else:
            l_parcial = text.split()[w*(length-overlap):w*(length-overlap) + length]

        l_total.append(" ".join(l_parcial))
    return l_total

def wrap_tokenizer(tokenizer, padding=True, truncation=True, return_tensors='pt', max_length=512):
    def tokenize(text):
        tokens = tokenizer(
            text,
            padding=padding, 
            return_attention_mask=True,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors
            )
        tokens=[tokens['input_ids'],tokens['attention_mask']]
        return tokens
    return tokenize

class WesBertModel(nn.Module):
    def __init__(self,tam):
        super(WesBertModel,self).__init__()
        self.bert=BertModel.from_pretrained('neuralmind/bert-base-portuguese-cased', output_hidden_states = True)
        self.bert.resize_token_embeddings(tam)
    def extract_emb(self,tokens):
        with torch.no_grad():
            outputs = self.bert(tokens[0], tokens[1])
            hidden_states = outputs[2]
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = token_embeddings.permute(1,2,0,3)
        token_vecs_sum = []
        for batch in token_embeddings:
            linha=[]
            for token in batch:
                sum_vec = torch.sum(token[-4:], dim=0)
                linha.append(sum_vec)
            token_vecs_sum.append(linha)
        lista=[]
        for cada in token_vecs_sum:
            lista.append(torch.stack(cada))
        return(torch.stack(lista))

def retorna_embeds(dataf):
    tokenizador = BertTokenizer.from_pretrained('./tokenizer-light')
    tokenize=wrap_tokenizer(tokenizador)
    model = WesBertModel(len(tokenizador))
    listdf=[]
    for indice,item in tqdm(dataf.iterrows(), total=dataf.shape[0]):
        tokens = tokenize(item['Text'])
        embeddings = model.extract_emb(tokens)

        pooled_emb = torch.mean(embeddings, axis=1)
        item['embeddings']=pooled_emb
        listdf.append(item)
    novodf=pd.DataFrame(listdf)
    novodf.to_pickle('../Datasets/multilabel-habilitacao-embeddings-.pkl')

start = timer() 
NUM_THREADS = 24
threads_criadas = []

for id_th in range(1,NUM_THREADS+1):
    nova_thread = multiprocessing.Process(target=self.all,args=(id_th,NUM_THREADS,nomes))
    threads_criadas.append(nova_thread)
    nova_thread.start()

for td in threads_criadas:
    td.join()

end = timer()
print("Tempo total de processamento: ",end - start)

def uneCsv():
    dataframes = []
    for arquivo in os.listdir():
        if arquivo.find('Dataframe') != -1:
            try:
                df_aux = pd.read_csv(arquivo, encoding=('utf-8'))
                dataframes.append(df_aux)
            except:
                continue
    df = pd.concat(dataframes, axis=0)
    df.to_csv('DFCompleto.csv', index=False)

multilabel=pd.read_csv('../Datasets/multilabel-habilitacao.csv')
multilabel.rename(columns={'text':'Text','classe':'Label'},inplace=True)

uneCsv()

#Estrategia para particionar um numero
multilabel=multilabel.sample(n=100,ignore_index=True)

multilabel.Text = multilabel.Text.apply(lambda x: get_text_split(x))
multilabel['n_chunks'] = multilabel.Text.apply(lambda x: len(x))
multilabel['embeddings']=np.nan
retorna_embeds(multilabel)