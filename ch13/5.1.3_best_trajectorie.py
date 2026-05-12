import pandas as pd

# 1. Define o nome do arquivo Parquet
file_path = 'trajectory_3_gram_1950_2010.parquet'

# 2. Carrega o arquivo Parquet
try:
    df_3gram = pd.read_parquet(file_path)
except FileNotFoundError:
    print(f"Erro: Arquivo Parquet não encontrado em {file_path}.")
    exit()

# 3. Calcula a Frequência Absoluta Total de cada termo
# (Soma da coluna 'Count' agrupada por 'Term')
term_ranking = df_3gram.groupby('Term')['Count'].sum().reset_index(name='Total_Absolute_Frequency')

# 4. Ordena e exibe os 10 termos mais frequentes
top_10_terms = term_ranking.sort_values(by='Total_Absolute_Frequency', ascending=False).head(10)

print("--- Top 10 Termos 3-Gram por Frequência Absoluta (1950-2010) ---")
print(top_10_terms)