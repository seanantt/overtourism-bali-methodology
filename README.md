# Overtourism in Bali – Methodology

Code, data, and trained model for the study on perceived overcrowding in Bali 
using BERT-based sentiment classification and spatial autocorrelation analysis.

## 📂 Dataset & Trained Model Access

Hosted on Google Drive due to GitHub file-size limits:  
[Google Drive folder](https://drive.google.com/drive/folders/1bhQX5fU_uv9-aiV4aWFHXsxi7UY0DI9c?usp=sharing)


This folder contains:
- `cleaned_data_reviews.xlsx` – Cleaned review dataset after text preprocessing  
- `predicted_review.xlsx` –  Predicted review dataset after text preprocessing  
- `crowded_uncrowded_model` – Fine-tuned BERT model directory  


## Parameters

**BERT fine-tuning**
* Backbone: bert-base-uncased
* Max sequence length: 128
* Training batch size: 8
* Evaluation batch size: 16
* Epochs: 3
* Optimizer: AdamW with 500 warm-up steps
* Weight decay: 0.01
* Train/test split: 80/20 with fixed random seed

**KDE (QGIS)**
* CRS: WGS 84 / UTM Zone 50S (EPSG:32750)
* Kernel: Gaussian
* Bandwidth: 2,000 m (crowding reviews); 1,500 m (infrastructure)
* Cell size: 100 m

**Spatial autocorrelation (GeoDa)**
* Aggregation: 1 km × 1 km grid, restricted to Bali land mask
* Spatial weights: row-standardized KNN, K = 5, Euclidean distance
* Global Moran's I, LISA, Bivariate Moran's I
* Sensitivity check: K ∈ {4, 6, 8}


## Software
QGIS, GeoDa, Python (Hugging Face Transformers).
