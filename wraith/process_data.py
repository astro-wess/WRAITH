import os
import re
import csv

def obter_valor_fracao(arquivo, prefixo, ocorrencia):
    with open(arquivo, 'r') as f:
        fracoes = [linha.split()[1] for linha in f if linha.startswith(prefixo)]
        if ocorrencia < len(fracoes):
            return fracoes[ocorrencia]
    return None

def obter_valor_criterio(arquivo, criterio):
    with open(arquivo, 'r') as f:
        for linha in f:
            if linha.startswith(f'#   {criterio}:'):
                valor_criterio = linha.split(': ')[1].strip()
                return valor_criterio
    return None

def adicionar_colunas_zeros(output_csv, novos_headers):
    # Carregar o CSV original
    with open(output_csv, 'r') as csvfile:
        reader = csv.reader(csvfile)
        linhas_csv = list(reader)

    # Adicionar novos cabeçalhos se necessário
    if not any(header in linhas_csv[0] for header in novos_headers):
        linhas_csv[0].extend(novos_headers)
        for linha in linhas_csv[1:]:
            linha.extend(['0'] * len(novos_headers))

    # Escrever de volta ao CSV com as novas colunas
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(linhas_csv)

def get_results(name, gal_name, gal_file, output_csv, components, filter):
    result_dir = f'./{name}/results'
    file_number = int(re.findall(r'\d+', gal_name)[0]) 
    arquivo_dat = f'{file_number}{filter}_best{components}comp.dat'  # O nome do arquivo específico

    caminho_arquivo = os.path.join(result_dir, arquivo_dat)
    if not os.path.exists(caminho_arquivo):
        print(f"Arquivo {caminho_arquivo} não encontrado.")
        return

    params = {
        "PA": [],
        "ell": [],
        "n": [],
        "r_e": [],
        "I_e": []
    }

    # Coletar os valores para duas rodadas
    for i in range(2):
        params["PA"].append(obter_valor_fracao(caminho_arquivo, 'PA', i))
        params["ell"].append(obter_valor_fracao(caminho_arquivo, 'ell', i))
        params["n"].append(obter_valor_fracao(caminho_arquivo, 'n', i))
        params["r_e"].append(obter_valor_fracao(caminho_arquivo, 'r_e', i))
        params["I_e"].append(obter_valor_fracao(caminho_arquivo, 'I_e', i))

    # Coletar valores dos critérios
    criterios = ["Reduced value", "AIC", "BIC"]
    valores_criterios = [obter_valor_criterio(caminho_arquivo, criterio) for criterio in criterios]

    # Adicionar colunas com valores zero
    novos_headers = ["Reduced value", "AIC", "BIC", "PA1", "ell1", "n1", "r_e1", "I_e1", "PA2", "ell2", "n2", "r_e2", "I_e2"]
    adicionar_colunas_zeros(output_csv, novos_headers)

    # Carregar o CSV original novamente para modificar a linha específica
    with open(output_csv, 'r') as csvfile:
        reader = csv.reader(csvfile)
        linhas_csv = list(reader)
    # Atualizar a linha correspondente
    if 1 <= file_number < len(linhas_csv):  # Garantir que o índice esteja dentro do intervalo correto
        # Preencher valores dos critérios
        start_col = len(linhas_csv[0]) - len(novos_headers)
        for i, criterio in enumerate(valores_criterios):
            linhas_csv[file_number - 1][start_col + i] = criterio if criterio is not None else '0'
        
        # Preencher valores das duas rodadas de parâmetros
        for i, key in enumerate(params):
            if len(params[key]) >= 2:  # Garantir que temos valores para ambas as rodadas
                if start_col + i < len(linhas_csv[file_number]):
                    linhas_csv[file_number -1][start_col + i + 3] = params[key][0] if params[key][0] is not None else '0'
                if start_col + i + 5 < len(linhas_csv[file_number]):
                    linhas_csv[file_number -1][start_col + i + 8] = params[key][1] if params[key][1] is not None else '0'

    # Escrever de volta ao CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(linhas_csv)
