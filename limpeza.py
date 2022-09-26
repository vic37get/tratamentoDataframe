
def limpezaDataframe(dataframe):
    dados = dataframe.replace('\\n\d{1,2}\.\d{0,2}\.?\d{0,2}\.?\d{0,2}\.?','',regex=True)
    dados = dados.replace('\n', ' ', regex=True)
    return dados
