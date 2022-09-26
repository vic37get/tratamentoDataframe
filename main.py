import alteraClasse as alt
import limpeza as limp
import removeTomadaPreco as rtp
import pandas as pd

# limpa Dataframe
#dataframe = pd.read_csv('DataframeConcat.csv')
#df = limp.limpezaDataframe(dataframe)
#df.to_csv('DataframeLimpo.csv')

#Altera a classe dos achados que estão fora das 5 seções
dataframe = pd.read_csv('DataframeLimpo.csv')
df = alt.alteraClasse(dataframe)
df.to_csv('DataframeLimpo5Secoes.csv')