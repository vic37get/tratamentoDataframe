#Altera a classe dos achados que estão fora das 5 seções
import re
import expressoesSecao as exp
global colunas
global bar
from tqdm import tqdm



def alteraClasse(dataframe):
    colunas = dataframe.columns
    dataframe.fillna('',inplace=True)
    bar = tqdm(total=dataframe.shape[0])
    dados = dataframe.apply(filter_replace,axis=1)
    return dados

def filter_replace(row):
    for indice, elemento in enumerate(row):
        nome_coluna = colunas[indice]
        if re.search(re.compile('SECAO'),nome_coluna):
            if not((row[indice] == '' or re.search(exp.OBJETO,row[indice]) or re.search(exp.JULGAMENTO,row[indice]) or re.search(exp.CONDICAO_PARTICIPACAO,row[indice]) or re.search(exp.HABILITACAO,row[indice]) or re.search(exp.CREDENCIAMENTO,row[indice]))):
                row[indice-1] = 0
    bar.update(1)
    return row