import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

class Perceptron:
    def __init__(self, input_size, taxa_aprendizado=0.01):
        self.w = np.random.rand(input_size + 1) # Inicialização dos pesos de forma aleatória
        self.N = taxa_aprendizado
        self.epocas = 0
        
    def funcao_ativacao(self, x):
        # Função de ativação do perceptron (degrau)
        return 1 if x >= 0 else -1
    
    def treinamento(self, x_treino, d_treino):
        while True:
            erro = False
            for i in range(len(x_treino)):
                # Adiciona o bias à amostra de treino
                xi = np.insert(x_treino[i], 0, -1)
                pot = np.dot(xi, self.w)
                y = self.funcao_ativacao(pot)
                if y != d_treino[i]:
                    self.w += self.N * (d_treino[i] - y) * xi
                    erro = True
            self.epocas += 1           
            if not erro:
                break
    
    def operacao(self, x_teste):
        teste = []
        for i in range(len(x_teste)):
            # Adiciona o bias à amostra de teste
            xi = np.insert(x_teste[i], 0, -1)
            pot = np.dot(xi, self.w)
            y = self.funcao_ativacao(pot)
            teste.append(y)
        return teste


def separar_dados(arquivo_xls, tamanho_teste=0.3, random_state=None):
    # Leitura do arquivo Excel para um DataFrame
    pdDataFrame = pd.read_excel(arquivo_xls)
    x = pdDataFrame.iloc[:, :-1].values
    d = pdDataFrame.iloc[:, -1].values    

    # Se um estado aleatório for especificado, define a semente
    if random_state is not None:
        np.random.seed(random_state)
    
    # Embaralha os índices dos dados
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    
    # Calcula o número de amostras para o conjunto de teste
    sample_testes = int(tamanho_teste * len(x))
    
    # Separa os índices para o conjunto de teste e treino
    indices_teste = indices[:sample_testes]
    indices_treino = indices[sample_testes:]
    
    # Cria os conjuntos de treino e teste
    x_treino, x_teste = x[indices_treino], x[indices_teste]
    d_treino, d_teste = d[indices_treino], d[indices_teste]
    
    return x_treino, x_teste, d_treino, d_teste

if __name__ == "__main__":
    arquivo_xls = "./Treinamento_Perceptron.xls"
    
    # Chama a função para separar os dados
    x_treino, x_teste, d_treino, d_teste = separar_dados(arquivo_xls)
    print(x_treino, x_teste, d_treino, d_teste)
    
    # Leitura do arquivo Excel para um DataFrame
    pdDataFrame = pd.read_excel(arquivo_xls)
    x = pdDataFrame.iloc[:, :-1].values
    d = pdDataFrame.iloc[:, -1].values  
    
    # Inicializa o perceptron com o tamanho da entrada
    perceptron = Perceptron(input_size=x_treino.shape[1])
    
    print(f"Pesos Iniciais: {perceptron.w}")
    
    # Treina o perceptron com os dados de treino
    perceptron.treinamento(x_treino, d_treino)
    
    # Exibe os pesos treinados e o número de épocas
    print(f"Pesos Treinados: {perceptron.w}")
    print(f"Épocas: {perceptron.epocas}")

    
    # Realiza a operação de teste e obtém os resultados
    r_teste = np.array(perceptron.operacao(x_teste))
    print(f"Resultados Teste: {r_teste}\n")
    print(f"Resultado Esperado: {d_teste}\n")
    
    # Calcula e exibe a matriz de confusão
    matriz_conf = confusion_matrix(d_teste, r_teste)
    matriz_conf_df = pd.DataFrame(matriz_conf, index=["Atual -1", "Atual 1"], columns=["Previsão -1", "Previsão 1"])
    print("\nMatriz de confusão (DataFrame):")
    print(matriz_conf_df)