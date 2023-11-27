# DC_CCN057_RedesNeuraisArtificiais

Este repositório contém implementações em Python baseadas no livro `Redes Neurais Artificiais Para Engenharia E Ciências Aplicadas. Curso Prático`
---


# Implementações do Perceptron e Adaline

Implementações em Python dos algoritmos Perceptron e Adaline para classificação binária. Os scripts utilizam NumPy para operações numéricas, pandas para manipulação de dados e scikit-learn para avaliar o desempenho do modelo.

## Perceptron

### Visão Geral
O Perceptron é um algoritmo simples de classificação binária que aprende uma fronteira de decisão linear. O script em Python fornecido (`perceptron.py`) inclui uma classe Perceptron com métodos para treinamento e teste. O script também contém uma função para separar os dados em conjuntos de treinamento e teste.

### Utilização
1. **Instale as Dependências:**
   ```bash
   pip install numpy pandas scikit-learn
   ```

2. **Execute o Script:**
   ```bash
   python perceptron.py
   ```
   Isso irá carregar os dados de "Treinamento_Perceptron.xls", treinar o Perceptron e imprimir os resultados, incluindo a matriz de confusão.

## Adaline

### Visão Geral
O algoritmo Adaline (Adaptive Linear Neuron) é uma melhoria sobre o Perceptron, utilizando uma função de ativação linear e minimizando o erro quadrático médio durante o treinamento. O script em Python fornecido (`adaline.py`) inclui uma classe Adaline com métodos para treinamento e teste. Ele também contém uma função para separar os dados em conjuntos de treinamento e teste.

### Utilização
1. **Instale as Dependências:**
   ```bash
   pip install numpy pandas scikit-learn
   ```

2. **Execute o Script:**
   ```bash
   python adaline.py
   ```
   Isso irá carregar os dados de treinamento de "Treinamento_Perceptron.xls" e os dados de teste de "Teste_Perceptron.xls", treinar o Adaline e imprimir os resultados, incluindo os pesos treinados e o número de épocas.

## Requisitos
- Python 3.x
- NumPy
- pandas
- scikit-learn
