import re

OBJETO = re.compile(r'((OBJETO)|(DA LICITA[ÇC][ÃAÂ]O))')
JULGAMENTO = re.compile(r'((JULGAMENTO)|(AN[AÁ]LISE[S]?.{0,20}PROPOSTA[S]?))')
CONDICAO_PARTICIPACAO = re.compile(r'(PARTICIP)')
HABILITACAO = re.compile(r'(HABILITA)')
CREDENCIAMENTO = re.compile(r'(CREDENCI)')
