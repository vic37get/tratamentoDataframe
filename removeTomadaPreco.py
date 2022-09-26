#Altera a classe dos achados Integralizado para 0 caso sejam tomada de preço
import numpy as np
global integralizado
global bar
from tqdm import tqdm

def filter_integralizado(row):
    integr_np = integralizado.values
    i = 0
    while(integr_np.shape[0] != 0):
        if(i>=integr_np.shape[0]):
            bar.update(1)
            return row
        id_licitacao_para_excluir = integr_np[i][0]
        id_arquivo_para_excluir = integr_np[i][1]
        if row['ID-LICITACAO'] == id_licitacao_para_excluir and row['ID-ARQUIVO'] == id_arquivo_para_excluir:
            row['integralizado'] = 0
            row['TXT_integralizado'] = ''
            row['SECAO_integralizado'] = float('Nan')
            integr_np = np.delete(integr_np,[i],0)
            i = 0
        else:
            i += 1
    bar.update(1)
    return row


def removeTomadaPreco(integralizado, dataframe):
    bar = tqdm(total=dataframe.shape[0])
    listaAseremTrocados = []
    
    for i in integralizado.itertuples():
        line = dataframe[(dataframe['ID-LICITACAO']==i.ID) & (dataframe['ID-ARQUIVO']==i.ID_ARQUIVO)]
        if(line.shape[0] == 1):
            if line.integralizado.values[0] == 1:
                listaAseremTrocados.append(line[['ID-LICITACAO','ID-ARQUIVO']].values[0])
        bar.update(1)

    for i in listaAseremTrocados:
        dataframe.loc[(dataframe['ID-LICITACAO']==i[0]) & (dataframe['ID-ARQUIVO']==i[1]),'integralizado'] = 0
        dataframe.loc[(dataframe['ID-LICITACAO']==i[0]) & (dataframe['ID-ARQUIVO']==i[1]),'TXT_integralizado'] = ''
        dataframe.loc[(dataframe['ID-LICITACAO']==i[0]) & (dataframe['ID-ARQUIVO']==i[1]),'SECAO_integralizado'] = ''

    return dataframe