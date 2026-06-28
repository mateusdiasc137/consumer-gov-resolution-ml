Os dados brutos devem ser baixados da página de dados abertos do Consumidor.gov.br.
Após o download, os arquivos CSV devem ser colocados em data/raw/.


Os arquivos processados podem ser recriados rodando:
python src/01_limpar_dados.py
python src/02_criar_conjuntos_modelagem.py
