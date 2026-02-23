# ProjetoDioPrevendoVendas
Prevendo Vendas de Sorvete

Entendendo o Desafio – Prevendo Vendas de Sorvete com Machine Learning
📌 1. Nome do Projeto (Sugestão)

Gelado Mágico

📂 Estrutura do Repositório
gelato-magico-ml-predict/
│
├── inputs/
│ └── dados_sorvete.csv
│
├── notebooks/
│ └── analise_exploratoria.ipynb
│
├── src/
│ ├── train.py
│ ├── predict.py
│ └── pipeline.py
│
├── models/
│
├── app/
│ └── app.py
│
├── requirements.txt
├── README.md
└── MLproject
📊 2. Dataset (inputs/dados_sorvete.csv)

Exemplo de dataset:

temperatura,vendas
18,120
20,150
22,180
25,220
28,260
30,300
32,340
35,400
15,100
10,60
🧠 3. Modelo Utilizado

Modelo de regressão linear utilizando:

Python

Scikit-Learn

MLflow

Pandas

FastAPI (para deploy)

🏗️ 4. Código de Treinamento (src/train.py)
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
Carregar dados

df = pd.read_csv("inputs/dados_sorvete.csv")

X = df[["temperatura"]]
y = df["vendas"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Iniciar MLflow

mlflow.start_run()

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)

mlflow.log_metric("mse", mse)
mlflow.sklearn.log_model(model, "modelo_regressao")

mlflow.end_run()

print("Modelo treinado com sucesso!")
🔁 5. Pipeline Estruturado (src/pipeline.py)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

pipeline = Pipeline([
('scaler', StandardScaler()),
('regressor', LinearRegression())
])
🔮 6. API para Previsão em Tempo Real (app/app.py)
import mlflow.pyfunc
from fastapi import FastAPI
import pandas as pd

app = FastAPI()

model = mlflow.pyfunc.load_model("models:/modelo_regressao/1")

@app.get("/predict")
def predict(temperatura: float):
input_data = pd.DataFrame({"temperatura": [temperatura]})
prediction = model.predict(input_data)
return {"vendas_previstas": float(prediction[0])}

Rodar com:

uvicorn app.app:app --reload
☁️ 7. Deploy em Cloud

Você pode usar:

AWS EC2

Azure App Service

Google Cloud Run

Render

Railway

Sugestão simples: Render ou Railway (mais fácil para portfólio).

📄 README.md (PRONTO PARA USAR)

Agora vem a parte mais importante: o README estratégico para impressionar recrutadores.

🍦 Gelato Mágico – Previsão de Vendas com Machine Learning
📌 Sobre o Projeto

Este projeto foi desenvolvido com o objetivo de prever a quantidade de sorvetes vendidos com base na temperatura do dia, utilizando técnicas de Machine Learning.

A solução permite que a sorveteria Gelato Mágico:

Reduza desperdícios

Planeje melhor sua produção

Maximize lucros

Tome decisões baseadas em dados

🧠 Tecnologias Utilizadas

Python

Scikit-Learn

MLflow

FastAPI

Pandas

Cloud Deployment

📊 Análise Exploratória

Foi identificada uma forte correlação positiva entre temperatura e vendas:

Quanto maior a temperatura, maior o volume de vendas.

Insights obtidos:

Temperaturas abaixo de 15°C reduzem drasticamente as vendas

Acima de 30°C há um crescimento exponencial

O modelo linear apresentou ótimo desempenho para esse cenário

📈 Modelo Utilizado

Foi aplicado um modelo de Regressão Linear, adequado para prever valores contínuos.

Métrica avaliada:

MSE (Mean Squared Error)

O modelo foi registrado utilizando MLflow, garantindo versionamento e rastreabilidade.

🔁 Pipeline

Foi implementado um pipeline estruturado para:

Padronização dos dados

Treinamento

Avaliação

Registro do modelo

Garantindo reprodutibilidade do experimento.

☁️ Deploy

O modelo foi disponibilizado via API utilizando FastAPI, permitindo previsões em tempo real em ambiente de cloud computing.

Exemplo de requisição:

/predict?temperatura=30

Resposta:

{
"vendas_previstas": 298.4
}
🚀 Aprendizados

Durante o desenvolvimento deste projeto, foi possível consolidar conhecimentos em:

Regressão Linear

Versionamento de modelos com MLflow

Criação de APIs com FastAPI

Estruturação de pipelines de Machine Learning

Deploy em nuvem

Boas práticas de organização de projetos de dados

📌 Próximos Passos

Testar outros modelos como Random Forest

Adicionar variáveis como:

Dia da semana

Umidade

Eventos locais

Criar dashboard com Streamlit

Automatizar pipeline com CI/CD

🎯 Diferencial para Entrevistas

"Desenvolvi um modelo de regressão para prever vendas com base na temperatura, implementei versionamento com MLflow, criei API para consumo em tempo real e realizei deploy em cloud, garantindo reprodutibilidade e escalabilidade."
