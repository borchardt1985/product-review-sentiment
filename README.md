# Análise de Sentimento de Avaliações de Produtos com Python e Machine Learning

---

## Visão Geral do Projeto

Este projeto implementa um sistema de **Análise de Sentimento** capaz de classificar avaliações de produtos como **positivo**, **negativo** ou **neutro**. Utilizando técnicas de Processamento de Linguagem Natural (PLN) e Machine Learning, o objetivo é demonstrar um pipeline completo desde a coleta e pré-processamento de dados textuais até o treinamento e a avaliação de um modelo de classificação.

### Aplicações Reais

A Análise de Sentimento é crucial em diversas áreas, como:
* **E-commerce:** Para entender a satisfação do cliente com produtos e serviços.
* **Atendimento ao Cliente:** Para monitorar o humor geral das interações e identificar problemas rapidamente.
* **Marketing:** Para avaliar campanhas e a percepção da marca.
* **Pesquisa de Mercado:** Para extrair insights sobre opiniões públicas e tendências.

---

## Tecnologias Utilizadas

* **Python:** Linguagem de programação principal.
* **Pandas:** Para manipulação e análise de dados tabulares.
* **NLTK (Natural Language Toolkit):** Para pré-processamento de texto (tokenização, remoção de stopwords).
* **Scikit-learn:** Para vetorização de texto (TF-IDF) e construção do modelo de Machine Learning (Regressão Logística).
* **Git & GitHub:** Para controle de versão e hospedagem do projeto.

---

## Como Rodar o Projeto (Passo a Passo)

Siga estas instruções para configurar e executar o projeto em sua máquina local:

### Pré-requisitos

Certifique-se de ter o **Python 3.x** e o **pip** (gerenciador de pacotes do Python) instalados em seu sistema.

### Instalação

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/borchardt1985/product-review-sentiment.git](https://github.com/borchardt1985/product-review-sentiment.git)
    cd product-review-sentiment
    ```
    *(Atenção: Altere `product-review-sentiment` para o nome do seu repositório se for diferente)*

2.  **Crie e ative um ambiente virtual:**
    É altamente recomendado usar um ambiente virtual para isolar as dependências do projeto.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # macOS/Linux
    # .\venv\Scripts\Activate.ps1  # Windows PowerShell
    # venv\Scripts\activate.bat   # Windows CMD
    ```

3.  **Instale as dependências Python:**
    ```bash
    pip install pandas scikit-learn nltk matplotlib seaborn
    ```

4.  **Baixe os recursos de dados do NLTK:**
    Abra o interpretador Python e execute os comandos:
    ```bash
    python
    >>> import nltk
    >>> nltk.download('punkt')
    >>> nltk.download('stopwords')
    >>> nltk.download('punkt_tab')
    >>> exit()
    ```

### Execução

1.  **Execute o script principal:**
    ```bash
    python data_exploration.py
    ```
    *(Você pode renomear `data_exploration.py` para `main.py` ou `sentiment_analyzer.py` para um nome mais direto se quiser, mas lembre-se de atualizar este README.)*

---

## Estrutura do Conjunto de Dados (`reviews.csv`)

O projeto utiliza um arquivo CSV simples (`reviews.csv`) com duas colunas: `review` (o texto da avaliação) e `sentiment` (o rótulo de sentimento: 'positivo', 'negativo', 'neutro').

Exemplo:
```csv
review,sentiment
"Este produto é excelente, amei!",positivo
"Péssima qualidade, não compraria novamente.",negativo
# ... e assim por diante
```

---

## Resultados do Modelo
O modelo foi treinado e avaliado usando um conjunto de dados de 229 avaliações. Os resultados são apresentados abaixo:

### Métricas de Avaliação:

```
--- Métricas de Avaliação do Modelo ---
Acurácia: 0.80

Relatório de Classificação:
              precision    recall  f1-score   support

    negativo       0.81      0.81      0.81        16
      neutro       0.92      0.73      0.81        15
    positivo       0.72      0.87      0.79        15

    accuracy                           0.80        46
   macro avg       0.82      0.80      0.81        46
weighted avg       0.82      0.80      0.81        46
```

### Matriz de Confusão
```
Matriz de Confusão:
 [[13  1  1]
  [ 3 13  0]
  [ 2  2 11]]
```
(Observação: A ordem das classes na matriz é 'positivo', 'negativo', 'neutro' conforme definido no código.)

### Previsões do Exemplo
Aqui estão algumas previsões do modelo em novas frases:
```
Avaliação: 'O serviço foi excelente e a entrega rápida. Recomendo muito!'
Sentimento Previsto: positivo

Avaliação: 'Produto terrível, dinheiro jogado fora, nunca mais compro.'
Sentimento Previsto: negativo

Avaliação: 'Achei o produto ok, nada de especial, cumpre o básico.'
Sentimento Previsto: neutro

Avaliação: 'Experiência incrível, superou todas as expectativas!'
Sentimento Previsto: positivo

Avaliação: 'Totalmente insatisfeito, quebrou na primeira semana.'
Sentimento Previsto: negativo
```

---

## Considerações e Limitações
Este projeto demonstra um pipeline completo de análise de sentimento. É importante notar que a **performance do modelo é altamente dependente da quantidade e qualidade dos dados de treinamento**. Com um dataset de 229 avaliações, o modelo já apresenta uma acurácia razoável (80%). No entanto, para aplicações em larga escala e maior confiabilidade, seria necessário treinar o modelo com milhares ou milhões de exemplos.

---

## Próximos Passos e Possíveis Melhorias
- Expandir o Dataset: Coletar e utilizar um volume muito maior de avaliações de produtos reais.

- Explorar Outros Modelos: Testar outros algoritmos de Machine Learning (e.g., Naive Bayes, SVM) ou técnicas mais avançadas de PLN como Deep Learning (Redes  Neurais Recorrentes, Transformers com bibliotecas como *Hugging Face*).

- Otimização de Hiperparâmetros: Ajustar os parâmetros do modelo para obter o melhor desempenho possível.

- Balanceamento de Classes: Se um dataset futuro for desbalanceado (muito mais de um tipo de sentimento), aplicar técnicas de balanceamento.

---

## Autor
Nelson Ariberto Borchardt - @borchardt1985

