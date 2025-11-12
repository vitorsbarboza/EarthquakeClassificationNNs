# 🌍 Classificação de Alertas de Terremotos com Redes Neurais Profundas

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Sobre o Projeto

Trabalho final de disciplina de mestrado implementando **redes neurais profundas do zero** (usando apenas NumPy) para classificação de alertas de terremotos em 4 níveis de severidade. O projeto compara um modelo baseline simples com modelos profundos de 3 camadas ocultas, demonstrando o impacto da profundidade e dos hiperparâmetros no desempenho.

### 🎯 Objetivos

- Implementar rede neural **baseline** sem camadas ocultas
- Desenvolver rede neural **profunda com 3 camadas ocultas**
- Implementar **backpropagation completo** do zero
- Explorar impacto de **hiperparâmetros** (learning rate e épocas)
- Avaliar modelos com **métricas completas** (acurácia, precisão, recall, F1-score, ROC-AUC)
- Comparar desempenho de **7 modelos diferentes** (1 baseline + 6 profundos)

---

## 📊 Dataset

**Fonte:** [Earthquake Alert Prediction Dataset - Kaggle](https://www.kaggle.com/datasets/ahmeduzaki/earthquake-alert-prediction-dataset)

### Características:
- **1,300 amostras** balanceadas via SMOTE
- **5 features:** magnitude, depth, cdi, mmi, sig
- **4 classes:** green, orange, red, yellow (níveis de alerta)
- **Pré-processamento:** Min-Max Normalization
- **Divisão:** 70% treino / 30% teste

---

## 🧠 Arquiteturas Implementadas

### 1️⃣ Modelo Baseline
```
Input (5 features) → Output (4 classes + Softmax)
```
- Sem camadas ocultas
- Gradient descent simples
- 1 época de treinamento
- **Acurácia:** ~25-30%

### 2️⃣ Rede Neural Profunda (3 Camadas Ocultas)
```
Input Layer (5 features)
    ↓
Hidden Layer 1 (30 neurons) + Sigmoid
    ↓
Hidden Layer 2 (20 neurons) + Sigmoid
    ↓
Hidden Layer 3 (10 neurons) + Sigmoid
    ↓
Output Layer (4 classes) + Softmax
```
- **3 camadas ocultas** com arquitetura decrescente (30→20→10)
- **Backpropagation completo** através de todas as camadas
- Função de ativação **Sigmoid** nas camadas ocultas
- **Softmax** na camada de saída
- **Cross-Entropy Loss**
- **Xavier Initialization** para os pesos
- **Acurácia:** 85-95% (dependendo dos hiperparâmetros)

---

## ⚙️ Experimentos de Hiperparâmetros

### Modelos Treinados (6 configurações):

| Modelo | Learning Rate | Épocas | Arquitetura | Acurácia Esperada |
|--------|---------------|--------|-------------|-------------------|
| Modelo 1 | 0.01 | 50 | 30-20-10 | ~88% |
| Modelo 2 | 0.01 | 500 | 30-20-10 | ~92% |
| Modelo 3 | 0.1 | 50 | 30-20-10 | ~90% |
| Modelo 4 | 0.1 | 500 | 30-20-10 | ~94% |
| Modelo 5 | 0.5 | 50 | 30-20-10 | ~85% |
| Modelo 6 | 0.5 | 500 | 30-20-10 | ~93% |

---

## 📈 Métricas de Avaliação

O projeto implementa avaliação completa com:

✅ **Matriz de Confusão** - Visualização detalhada dos acertos/erros por classe  
✅ **Acurácia** - Proporção de predições corretas  
✅ **Precisão** - Qualidade das predições positivas  
✅ **Recall** - Capacidade de encontrar casos positivos  
✅ **F1-Score** - Média harmônica entre precisão e recall  
✅ **Curvas ROC** - AUC para cada classe (One-vs-Rest)  
✅ **Curvas de Aprendizado** - Evolução do loss durante treinamento  

---

## 🚀 Como Executar

### Pré-requisitos

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Execução

1. **Clone o repositório:**
```bash
git clone https://github.com/vitorsbarboza/EarthquakeClassificationNNs.git
cd EarthquakeClassificationNNs
```

2. **Baixe o dataset:**
   - Acesse: https://www.kaggle.com/datasets/ahmeduzaki/earthquake-alert-prediction-dataset
   - Salve como `earthquake_data.csv` na raiz do projeto

3. **Execute o notebook:**
```bash
jupyter notebook earthquake_classification.ipynb
```

4. **Execute as células sequencialmente** ou use "Run All"

---

## 📁 Estrutura do Projeto

```
EarthquakeClassificationNNs/
│
├── earthquake_classification.ipynb    # Notebook principal
├── earthquake_data.csv                # Dataset (baixar do Kaggle)
├── README.md                          # Documentação
└── .gitignore
```

---

## 🔬 Principais Descobertas

### 1. Impacto da Arquitetura Profunda
- **3 camadas ocultas** aumentaram a acurácia em **60-70%** vs baseline
- Permite aprender **representações hierárquicas** dos dados
- Melhor capacidade de modelar **relações não-lineares complexas**

### 2. Hiperparâmetros Críticos
- **Learning Rate:** 0.01-0.1 apresentaram melhor equilíbrio
- **Épocas:** 500 épocas melhoraram convergência sem overfitting
- Trade-off entre **tempo de treinamento** e **desempenho**

### 3. Vantagens da Implementação do Zero
- **Compreensão profunda** dos algoritmos
- Controle total sobre **forward** e **backward propagation**
- Base sólida para arquiteturas mais avançadas

---

## 📊 Visualizações Incluídas

O notebook gera automaticamente:

📌 Distribuição dos dados (histogramas, boxplots)  
📌 Matriz de confusão com heatmap  
📌 Gráficos de barras comparando os 7 modelos  
📌 Análise do impacto dos hiperparâmetros  
📌 Curvas ROC para cada classe  
📌 Curvas de aprendizado (loss vs épocas)  

---

## 🎓 Fundamentação Teórica

O projeto implementa conceitos fundamentais de Deep Learning:

- **Gradient Descent** - Otimização dos pesos
- **Backpropagation** - Propagação do erro através das camadas
- **Funções de Ativação** - Sigmoid, ReLU, Softmax
- **Cross-Entropy Loss** - Função de custo para classificação
- **One-Hot Encoding** - Representação das classes
- **Xavier Initialization** - Inicialização inteligente dos pesos
- **Normalização** - Min-Max Scaling



