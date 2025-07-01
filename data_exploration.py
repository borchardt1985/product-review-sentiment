import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split # Nova importação
from sklearn.linear_model import LogisticRegression # Nova importação
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # Novas importações para avaliação

# 1. Carrega o arquivo CSV em um DataFrame do pandas
df = pd.read_csv('reviews.csv')

print("--- Primeiras 5 linhas do DataFrame Original ---")
print(df.head())
print("\n" + "-"*40 + "\n")

# 2. Configura as stop words (palavras a serem removidas) para português
try:
    stop_words = set(stopwords.words('portuguese'))
except LookupError:
    print("Erro: O pacote 'stopwords' para português não foi encontrado. Tentando 'english'.")
    stop_words = set(stopwords.words('english'))


# 3. Define a função de pré-processamento
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# 4. Aplica a função de pré-processamento à coluna 'review' e cria uma nova coluna 'processed_review'
df['processed_review'] = df['review'].apply(preprocess_text)

print("--- Avaliações Originais vs. Processadas ---")
print(df[['review', 'processed_review']].head())
print("\n" + "-"*40 + "\n")

# 5. Vetorização do Texto usando TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['processed_review'])
y = df['sentiment']

print("--- Dimensões da Matriz de Features (X) ---")
print(f"Número de avaliações: {X.shape[0]}")
print(f"Número de features (palavras): {X.shape[1]}")
print("\n" + "-"*40 + "\n")

# 6. Contagem de Sentimentos (mantido do passo anterior)
print("--- Contagem de Sentimentos ---")
print(df['sentiment'].value_counts())
print("\n" + "-"*40 + "\n")

# NOVOS CÓDIGOS A PARTIR DAQUI

# 7. Dividir os Dados em Treino e Teste
# test_size=0.2 significa que 20% dos dados serão usados para teste e 80% para treino.
# random_state=42 garante que a divisão seja a mesma cada vez que você rodar o código (reprodutibilidade).
# stratify=y garante que a proporção de cada sentimento seja mantida tanto no treino quanto no teste,
# o que é importante para datasets pequenos ou desbalanceados.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("--- Dimensões dos Dados de Treino e Teste ---")
print(f"Dimensões X_train: {X_train.shape}")
print(f"Dimensões X_test: {X_test.shape}")
print(f"Dimensões y_train: {y_train.shape}")
print(f"Dimensões y_test: {y_test.shape}")
print("\n" + "-"*40 + "\n")

# 8. Treinar o Modelo de Classificação
# Usaremos a Regressão Logística, que é um bom ponto de partida para classificação de texto.
# max_iter=1000 ajuda o algoritmo a convergir (encontrar a melhor solução) em mais iterações,
# útil para evitar avisos em datasets menores.
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train) # O modelo "aprende" com os dados de treino (X_train e y_train)

print("--- Modelo Treinado com Sucesso! ---")
print("\n" + "-"*40 + "\n")

# 9. Avaliar o Modelo
# Faz previsões nos dados de teste (X_test), que o modelo nunca viu antes.
y_pred = model.predict(X_test)

print("--- Métricas de Avaliação do Modelo ---")
# Acurácia: Porcentagem de previsões corretas.
print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}") # .2f para formatar com 2 casas decimais

# Relatório de Classificação: Mostra Precisão, Recall e F1-Score para cada classe (positivo, negativo, neutro).
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

# Matriz de Confusão: Tabela que mostra o número de previsões corretas e incorretas por classe.
# As linhas representam os valores reais, as colunas representam os valores previstos.
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred, labels=['positivo', 'negativo', 'neutro']))
# Especificamos 'labels' para garantir a ordem correta na matriz

# NOVOS CÓDIGOS A PARTIR DAQUI

# 10. Função para Prever Sentimento de Novas Avaliações
def predict_sentiment(text, vectorizer_obj, model_obj, stop_words_set):
    # Primeiro, pré-processa o texto da nova avaliação
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words_set]
    processed_text = ' '.join(tokens)

    # Em seguida, vetoriza o texto usando o MESMO vetorizador que foi treinado
    # Importante: usa .transform() e não .fit_transform() aqui,
    # pois o vetorizador já "aprendeu" o vocabulário.
    text_vectorized = vectorizer_obj.transform([processed_text])

    # Finalmente, o modelo faz a previsão
    prediction = model_obj.predict(text_vectorized)
    return prediction[0]

print("\n" + "="*50)
print("--- Testando o Modelo com Novas Avaliações ---")
print("="*50 + "\n")

# Exemplos de novas avaliações para testar
new_reviews_to_test = [
    "O serviço foi excelente e a entrega rápida. Recomendo muito!",
    "Produto terrível, dinheiro jogado fora, nunca mais compro.",
    "Achei o produto ok, nada de especial, cumpre o básico.",
    "Experiência incrível, superou todas as expectativas!",
    "Totalmente insatisfeito, quebrou na primeira semana."
]

for review in new_reviews_to_test:
    # Passa o vetorizador e o modelo que já treinamos para a função de previsão
    predicted_sentiment = predict_sentiment(review, vectorizer, model, stop_words)
    print(f"Avaliação: '{review}'")
    print(f"Sentimento Previsto: {predicted_sentiment}\n")

print("="*50)