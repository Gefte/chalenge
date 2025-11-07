# API de Predição de Crédito

Uma API simples e eficiente para predição de aprovação de crédito usando modelos de Machine Learning.

## Funcionalidades

- **Predição de Crédito**: Analisa características e retorna probabilidade de aprovação
- **Múltiplos Modelos**: Suporta Regressão Logística, Random Forest e Gradient Boosting
- **Confiança**: Indica níveis de confiança (Alto, Médio, Baixo)
- **Recomendações**: Fornece recomendações automáticas baseadas na predição

## Como Executar

### 1. Instalar Dependências

```bash
cd app
pip install -r requirements.txt
```

### 2. Iniciar a API

```bash
python main.py
```

A API estará disponível em: **http://localhost:8080**

### 3. Acessar Documentação

- **Interface Interativa**: http://localhost:8080
- **Documentação ReDoc**: http://localhost:8080/docs

## Endpoints

### `GET /health`
Verifica o status da API e modelos carregados.

**Resposta:**
```json
{
  "message": "API funcionando",
  "models_loaded": 3,
  "available_models": ["logistic_regression", "random_forest", "gradient_boosting"]
}
```

### `GET /models`
Lista todos os modelos disponíveis.

**Resposta:**
```json
{
  "models": [
    {"key": "logistic_regression", "name": "Regressão Logística"},
    {"key": "random_forest", "name": "Random Forest"},
    {"key": "gradient_boosting", "name": "Gradient Boosting"}
  ]
}
```

### `POST /predict`
Realiza predição de crédito.

**Entrada:**
```json
{
  "features": [0.5, 1.2, -0.3, 2.1, 0.8, 1.5],
  "model_name": "logistic_regression"
}
```

**Resposta:**
```json
{
  "prediction": 0.75,
  "probability": 0.82,
  "confidence": "Alto",
  "model_used": "Regressão Logística",
  "recommendation": "Aprovação recomendada"
}
```

## Testando a API

### Usando curl

```bash
curl http://localhost:8080/health

curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, 1.2, -0.3, 2.1, 0.8, 1.5]}'
```

### Usando Python

```python
import requests

response = requests.post(
    "http://localhost:8080/predict",
    json={"features": [0.5, 1.2, -0.3, 2.1, 0.8, 1.5]}
)
print(response.json())
```

## 🐳 Docker (Opcional)

### Construir Imagem

```bash
docker build -t credit-api .
```

### Executar Container

```bash
docker run -p 8080:8080 credit-api
```

## Estrutura dos Arquivos

```
app/
├── main.py              # Código principal da API
├── requirements.txt     # Dependências Python
├── Dockerfile          # Configuração Docker
├── logistic_regression.joblib  # Modelo treinado
├── random_forest.joblib        # Modelo treinado  
├── gradient_boosting.joblib    # Modelo treinado
└── scaler.joblib              # Normalizador
```

## 🔧 Configuração

A API pode ser configurada através de variáveis de ambiente:

- `HOST`: Endereço do servidor (padrão: 0.0.0.0)
- `PORT`: Porta do servidor (padrão: 8080)

Exemplo:
```bash
export HOST=127.0.0.1
export PORT=3000
python main.py
```

## 📋 Requisitos

- Python 3.8+
- FastAPI
- scikit-learn
- joblib
- numpy
- uvicorn

