import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

class Perceptron:
    def __init__(self, input_size, taxa_aprendizado=0.01):
        self.w = np.random.rand(input_size + 1) # Inicialização dos pesos de forma aleatória
        self.N = taxa_aprendizado
        self.epocas = 0
        
    def funcao_ativacao(self, x):
        # Função de ativação do perceptron (degrau bipolar)
        return 1 if x >= 0 else -1
    
    def treinamento(self, x_treino, d_treino):
        while True:
            erro = False
            for i in range(len(x_treino)):
                # Adiciona o limiar de ativação à amostra de treino
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
            # Adiciona o limiar de ativação à amostra de teste
            xi = np.insert(x_teste[i], 0, -1)
            pot = np.dot(xi, self.w)
            y = self.funcao_ativacao(pot)
            teste.append(y)
        return teste

if __name__ == "__main__":
    arquivo_xls_treino = "./Treinamento_Perceptron.xls"
    arquivo_xls_teste = "./Teste_Perceptron.xls"
    
    # Leitura do arquivo Excel para um DataFrame
    pdDataFrame = pd.read_excel(arquivo_xls_treino)
    print(pdDataFrame)
    x = pdDataFrame.iloc[:, :-1].values
    d = pdDataFrame.iloc[:, -1].values  
    
    pdDataFrameTeste = pd.read_excel(arquivo_xls_teste)
    x_teste = pdDataFrameTeste.iloc[:, :].values
    
    # Inicializa o perceptron com o tamanho da entrada
    perceptron = Perceptron(input_size=x.shape[1])
    
    print(f"Pesos Iniciais: {perceptron.w}")
    
    # Treina o perceptron com os dados de treino
    perceptron.treinamento(x, d)
    
    # Exibe os pesos treinados e o número de épocas
    print(f"Pesos Treinados: {perceptron.w}")
    print(f"Épocas: {perceptron.epocas}")
    
    print(f"Testes utilizando as 10 amostras do livro: {perceptron.operacao(x_teste)}")
