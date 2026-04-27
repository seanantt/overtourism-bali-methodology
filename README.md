# Overtourism in Bali – Methodology

The methodology includes data collection, text classification using BERT, spatial analysis with Kernel Density Estimation, and Pearson correlation to analyze the relationship between infrastructure and perceived overcrowding in Bali.

## 📘 Overview
The research aims to analyze perceived overcrowding in Bali's top tourist destinations using sentiment analysis on user reviews (TripAdvisor) and geospatial infrastructure mapping (OpenStreetMap). The methodology combines text classification (BERT-based), Kernel Density Estimation (KDE), and spatial correlation techniques to detect overtourism patterns.

## 📂 Dataset & Trained Model Access

Due to GitHub’s file size limitations, the full dataset and trained classification model are stored externally.

🔗 **You can access them here:**  
👉 [Google Drive – Dataset & Model Folder](https://drive.google.com/drive/folders/1bhQX5fU_uv9-aiV4aWFHXsxi7UY0DI9c?usp=sharing)

This folder contains:
- `cleaned_data_reviews.xlsx` – Cleaned review dataset after text preprocessing  
- `predicted_review.xlsx` –  Predicted review dataset after text preprocessing  
- `crowded_uncrowded_model` – Fine-tuned BERT model directory  


## Methodology Workflow

The study follows these steps:

1. Data collection
   - TripAdvisor reviews were scraped and georeferenced
   - Tourism infrastructure data was obtained from OpenStreetMap

2. Text preprocessing
   - Cleaning, tokenization, stopword removal

3. Text classification
   - Fine-tuned BERT model used to classify reviews into "crowded" and "uncrowded"

4. Spatial analysis
   - KDE performed in QGIS (EPSG:32750)
   - Radius:
     - Crowding reviews: 5000 m
     - Infrastructure: 5000 m
   - Cell size: 100 m

5. Spatial statistics
   - Global Moran’s I (ArcGIS)
   - LISA (ArcGIS)


## Reproducibility

To reproduce the results:

1. Download dataset from Google Drive
2. Load cleaned_data_reviews.xlsx
3. Run classification model from /crowded_uncrowded_model
4. Generate predicted labels (predicted_review.xlsx)
5. Import data into QGIS
6. Reproject to EPSG:32750
7. Run Heatmap (KDE) with:
   - Radius: 2000 meters
   - Cell size: 100 meters
8. Perform spatial autocorrelation in GeoDa



## Software
- QGIS 
- ArcGIS Pro 
- Python (Transformers / BERT)



