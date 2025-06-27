# Credit Risk Probability Model

## Created by: Addisu Taye  
**Date:** 2025-04-05  
**Project Description:** End-to-end implementation of a credit risk scoring model using alternative behavioral data.

Includes:
- Feature Engineering (RFM, Aggregate Features)
- Proxy Target Variable Creation (KMeans Clustering)
- Model Training & Evaluation (Logistic Regression, Random Forest, GBM)
- Deployment via FastAPI
- CI/CD Pipeline via GitHub Actions

# Credit Risk Probability Model

This project implements an end-to-end credit risk scoring model using alternative data sources. It transforms behavioral data into predictive risk signals to enable buy-now-pay-later services.

## Credit Scoring Business Understanding

### Basel II Accord Influence on Model Requirements
The Basel II Capital Accord emphasizes the importance of accurate risk measurement and capital adequacy. This influences our model requirements in several ways:

1. **Interpretability**: Financial institutions must be able to explain their credit decisions, especially when rejecting applicants. This pushes us toward models that balance performance with interpretability.

2. **Documentation**: The Basel framework requires thorough documentation of risk assessment processes, driving our focus on reproducible pipelines and experiment tracking.

3. **Risk Sensitivity**: The accord requires capital reserves to reflect actual risk levels, necessitating models that provide accurate probability estimates rather than just classifications.

### Proxy Variable Creation Necessity

Since we lack direct default information, creating a proxy variable is necessary because:

1. We need a target variable to supervise our machine learning model training
2. Real default data would require waiting for loan outcomes over time
3. Historical transaction patterns can serve as a reasonable proxy for future creditworthiness

Potential business risks of using this proxy include:

1. **Misclassification**: Customers might be incorrectly labeled as high/low risk
2. **Bias**: The proxy might capture behaviors not truly indicative of creditworthiness
3. **Regulatory challenges**: Proxies may not meet regulatory definitions of credit risk
4. **Performance drift**: Behavioral patterns may change over time reducing proxy validity

### Trade-offs: Simple vs Complex Models

| Consideration              | Simple Models (Logistic Regression/WoE) | Complex Models (Gradient Boosting) |
|---------------------------|------------------------------------------|------------------------------------|
| Interpretability          | High - easy to explain                   | Low - "black box" nature           |
| Regulatory Compliance     | Easier to satisfy                        | More challenging                   |
| Performance               | Good baseline                            | Generally better                   |
| Development Time          | Faster                                   | Longer                             |
| Monitoring Requirements   | Less intensive                           | More intensive                     |
| Feature Engineering Needs | Requires manual selection                | Can handle more features automatically |
| Stability                 | More stable                              | Potential for overfitting          |

## Getting Started

### Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Start the service: `docker-compose up`

### Endpoints

- `/`: Health check endpoint
- `/predict`: POST endpoint for credit risk prediction

### Usage Example

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "transactions": [
    {
      "Amount": 1000,
      "Value": 1000,
      "CountryCode": 256,
      "ProviderId": "ProviderId_6",
      "PricingStrategy": "2",
      "ProductCategory": "airtime",
      "ChannelId": "ChannelId_3",
      "TransactionStartTime": "2023-01-15T02:18:49Z"
    }
  ]
}'

Model Training
To train the model:

bash


1
python -m src.train
Testing
Run unit tests:

bash


1
pytest tests/
Linting
Check code style:


### 13. `.github/workflows/ci.yml`

```yaml
name: CI Pipeline

on:
  push:
    branches:
      - main

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v2
      
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        
    - name: Run linter
      run: |
        flake8 .
        
    - name: Run tests
      run: |
        pytest tests/
### 13. `.github/workflows/ci.yml`

```yaml
name: CI Pipeline

on:
  push:
    branches:
      - main

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v2
      
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        
    - name: Run linter
      run: |
        flake8 .
        
    - name: Run tests
      run: |