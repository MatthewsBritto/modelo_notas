from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import joblib

# Criar instância FastApi
app = FastAPI()

# Criar classe que terá os dados do request body para a API
class request_body(BaseModel):
  horas_estudo: float

# Carregar o modelo para realizar a predição
reg_model = joblib.load('./modelo_pontuacao.pkl')

@app.post('/predict')
def predict(data : request_body):
  # Preparar os dados para predição
  input_feature = [[data.horas_estudo]]

  # Fazer a predição
  prediction = reg_model.predict(input_feature)[0].astype(int)

  return {
    'pontuacao_teste': prediction.tolist()
  }