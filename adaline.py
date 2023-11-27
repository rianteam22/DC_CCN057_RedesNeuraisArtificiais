import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

class Adaline:
    def __init__(self, input_size, taxa_aprendizado=0.01, epslon = 0.002):
        self.w = np.random.rand(input_size + 1) # Inicialização dos pesos de forma aleatória
        self.N = taxa_aprendizado
        self.Epslon = epslon
        self.epocas = 0
        
    def funcao_ativacao(self, x):
        # Função de ativação do adaline (degrau bipolar)
        return 1 if x >= 0 else -1
    
    def erro_qm(self, x_treino, d_treino):
        eqm = 0
        for i in range(len(x_treino)):
            xi = np.insert(x_treino[i], 0, -1)
            pot = np.dot(xi, self.w)
            eqm += d_treino[i] - pot
        return (eqm/len(x_treino))
    
    def treinamento(self, x_treino, d_treino):
        while True:
            eqm_ant = self.erro_qm(x_treino, d_treino)
            print(f"Erro quadratico medio anterior: {eqm_ant}")
            for i in range(len(x_treino)):
                # Adiciona o limiar de ativação à amostra de treino
                xi = np.insert(x_treino[i], 0, -1)
                pot = np.dot(xi, self.w)
                self.w += self.N * (d_treino[i] - pot) * xi
            self.epocas += 1   
            eqm_atual = self.erro_qm(x_treino , d_treino)  
            print(f"Erro quadratico medio atual: {eqm_atual}")    
            if abs(eqm_atual - eqm_ant) <= self.Epslon:
                print(f"Erro eqm_atual - eqm_ant: {eqm_atual - eqm_ant}")  
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
    
    # Inicializa o adaline com o tamanho da entrada
    adaline = Adaline(input_size=x.shape[1])
    
    print(f"Pesos Iniciais: {adaline.w}")
    
    # Treina o adaline com os dados de treino
    adaline.treinamento(x, d)
    
    # Exibe os pesos treinados e o número de épocas
    print(f"Pesos Treinados: {adaline.w}")
    print(f"Épocas: {adaline.epocas}")
    
    print(f"Testes utilizando as 10 amostras do livro: {adaline.operacao(x_teste)}")
    

