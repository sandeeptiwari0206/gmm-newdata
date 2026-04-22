<div align="center">

# 🤖 GMM Product Recommendation System

### Unsupervised Machine Learning — Customer Segmentation & Personalised Recommendations on AWS SageMaker

[![Python](https://img.shields.io/badge/Python-Machine%20Learning-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com/sagemaker/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-GMM-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![S3](https://img.shields.io/badge/AWS-S3-569A31?style=for-the-badge&logo=amazons3&logoColor=white)](https://aws.amazon.com/s3/)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [How GMM Works](#-how-gmm-works)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Dataset & Outputs](#-dataset--outputs)
- [Getting Started](#-getting-started)
- [Deployment on AWS SageMaker](#-deployment-on-aws-sagemaker)
- [Pipeline Commands Reference](#-pipeline-commands-reference)
- [Output Files](#-output-files)
- [Author](#-author)

---

## 📖 Overview

This project implements a **Gaussian Mixture Model (GMM)** based product recommendation engine that:

1. **Segments customers** into behavioural clusters using unsupervised learning
2. **Analyses purchase patterns** within each cluster
3. **Generates personalised product recommendations** for every customer at scale
4. **Runs as a fully automated SageMaker Pipeline** on AWS — from raw data ingestion to recommendation CSV output

GMM is a probabilistic model that assumes data is generated from a mixture of several Gaussian distributions. Unlike K-Means (hard clustering), GMM assigns **soft probabilities** to each customer for each cluster — giving a richer and more nuanced segmentation.

---

## 🧠 How GMM Works

```
Raw Customer Data
       │
       ▼
┌─────────────────────────────────────────────────────┐
│              Data Preprocessing                      │
│  • Normalisation / Scaling                          │
│  • Feature Engineering (purchase frequency,         │
│    recency, category preferences, spend)            │
└─────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│         Gaussian Mixture Model (GMM)                 │
│                                                     │
│  • Fits N Gaussian distributions to the data        │
│  • EM Algorithm (Expectation-Maximisation)          │
│  • Each customer gets a probability for each        │
│    cluster → soft cluster assignment                │
│  • Optimal N selected via BIC / AIC score           │
└─────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│           Cluster Analysis & Profiling               │
│  • Identify top products per cluster                │
│  • Compute cluster centroids & characteristics      │
│  • Generate cluster summary report                  │
└─────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│         Recommendation Generation                    │
│  • For each customer → find their cluster           │
│  • Recommend top-N products from that cluster       │
│  • Priority-weighted & balanced outputs             │
│  • Export: all_customer_recommendations.csv         │
└─────────────────────────────────────────────────────┘
```

---

## 🏗 Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      AWS Cloud                                  │
│                                                                │
│   ┌──────────────┐        ┌──────────────────────────────┐    │
│   │   S3 Bucket  │        │     SageMaker Pipeline        │    │
│   │              │        │                              │    │
│   │  /data/      │──────► │  Step 1: Data Processing     │    │
│   │  /models/    │        │  Step 2: GMM Training        │    │
│   │  /output/    │◄───────│  Step 3: Cluster Profiling   │    │
│   │              │        │  Step 4: Recommendation Gen  │    │
│   └──────────────┘        └──────────────────────────────┘    │
│                                        │                       │
│                            ┌───────────▼──────────┐           │
│                            │  SageMaker Studio    │           │
│                            │  (Jupyter / Terminal) │           │
│                            └──────────────────────┘           │
└────────────────────────────────────────────────────────────────┘
```

**S3 Bucket:** `gmm-recommendation-bucket`
**Output Path:** `s3://gmm-recommendation-bucket/gmm-recommendation/recommendations/`

---

## 📁 Project Structure

```
gmm-newdata/
│
├── GMM_with_new_data.zip            # Main project source (extract to run)
│   └── GMM-custom/
│       ├── config/
│       │   └── config.yaml          # S3 bucket, region, model params
│       ├── scripts/
│       │   ├── create_pipeline.py   # Main pipeline orchestrator
│       │   └── check_pipeline_logs.py  # Monitor pipeline execution
│       ├── data/                    # Input CSVs (uploaded to S3)
│       └── requirements.txt
│
├── Bucket.zip                       # S3 bucket structure / config
│
├── GMM Block Digram Basic.drawio.png      # High-level architecture diagram
├── GMM Block Digram.drawio.png            # Full system block diagram
├── Gmm Block Digram Detailed.drawio.png   # Detailed component diagram
│
├── GMM_SageMaker_Guide_Diagrams_Moved.docx  # Full deployment guide
├── Clusters Summary.docx                    # Cluster analysis report (Hindi)
├── Clusters_Summary_English.docx            # Cluster analysis report (English)
│
├── recommandation balanced.csv              # Balanced recommendations output
├── recommendation by GMM on Lavi data.csv  # Recommendations on Lavi dataset
├── recommendation on priority.csv          # Priority-ranked recommendations
│
├── commands used.txt                # SageMaker terminal history
└── steps of deployment.txt          # 8-step deployment guide
```

---

## 🛠 Tech Stack

| Category | Technology |
|----------|-----------|
| **ML Algorithm** | Gaussian Mixture Model (GMM) |
| **ML Framework** | Scikit-learn |
| **Data Processing** | Python, Pandas, NumPy |
| **Cloud Platform** | AWS SageMaker Pipelines |
| **Storage** | AWS S3 |
| **SDK** | boto3, sagemaker Python SDK |
| **Environment** | SageMaker Studio (JupyterLab) |
| **Diagrams** | draw.io |

---

## 📊 Dataset & Outputs

### Input Data
Customer transaction data containing purchase history, product categories, frequency, recency, and spend — uploaded to S3 before pipeline execution.

### Output CSV Files

| File | Description |
|------|-------------|
| `recommendation by GMM on Lavi data.csv` | Raw recommendations generated on the Lavi customer dataset |
| `recommandation balanced.csv` | Balanced recommendations ensuring product diversity per customer |
| `recommendation on priority.csv` | Priority-ranked recommendations sorted by confidence score |

### Cluster Summary
Detailed cluster profiles are documented in:
- `Clusters_Summary_English.docx` — English version
- `Clusters Summary.docx` — Hindi version

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- AWS Account with SageMaker & S3 access
- IAM Role with `AmazonSageMakerFullAccess` and `AmazonS3FullAccess`

### 1. Clone & Extract

```bash
git clone https://github.com/sandeeptiwari0206/gmm-newdata.git
cd gmm-newdata
unzip GMM_with_new_data.zip
mv GMM-custom ~/
cd ~/GMM-custom
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install boto3 sagemaker
```

### 3. Configure

Edit `config/config.yaml` to set your S3 bucket name, AWS region, and model hyperparameters:

```yaml
# config/config.yaml
s3_bucket: gmm-recommendation-bucket
region: ap-south-1        # or your preferred region
n_components: 5           # number of GMM clusters
covariance_type: full
max_iter: 200
```

---

## ☁️ Deployment on AWS SageMaker

### 8-Step Deployment Guide

```
Step 1 → Verify AWS credentials & confirm S3 bucket exists
          aws s3 ls s3://gmm-recommendation-bucket

Step 2 → Install dependencies
          pip install sagemaker boto3

Step 3 → Upload input data to S3
          python scripts/create_pipeline.py --upload-data

Step 4 → Create the SageMaker Pipeline
          python scripts/create_pipeline.py --upsert

Step 5 → Run the pipeline
          python scripts/create_pipeline.py --upsert --start

Step 6 → Monitor execution
          python scripts/check_pipeline_logs.py
          # Or: AWS Console → SageMaker → Pipelines

Step 7 → Download recommendations output
          aws s3 cp s3://gmm-recommendation-bucket/gmm-recommendation/\
          recommendations/all_customer_recommendations.csv ./

Step 8 → (Optional) Deploy real-time inference endpoint
          python scripts/create_pipeline.py --deploy-endpoint
```

### Monitor in AWS Console

Navigate to: **AWS Console → SageMaker → Pipelines → gmm-recommendation-pipeline**

You'll see each pipeline step's status, logs, and execution time in real time.

---

## 📋 Pipeline Commands Reference

| Command | Description |
|---------|-------------|
| `python scripts/create_pipeline.py --upload-data` | Upload the 3 input CSVs to S3 |
| `python scripts/create_pipeline.py --upsert` | Create / update the pipeline definition |
| `python scripts/create_pipeline.py --upsert --start` | Create and immediately trigger a run |
| `python scripts/check_pipeline_logs.py` | Poll and display pipeline execution logs |
| `aws s3 cp s3://.../recommendations/... ./` | Download output recommendations locally |

---

## 📂 Output Files

After the pipeline completes, output files are available at:

```
s3://gmm-recommendation-bucket/
└── gmm-recommendation/
    ├── models/
    │   └── gmm_model.pkl          # Trained GMM model artifact
    ├── clusters/
    │   └── cluster_profiles.json  # Cluster analysis & top products
    └── recommendations/
        └── all_customer_recommendations.csv  # Final output
```

Download with:

```bash
aws s3 cp s3://gmm-recommendation-bucket/gmm-recommendation/recommendations/all_customer_recommendations.csv ./
```

---

## 👨‍💻 Author

<div align="center">

**Sandeep Tiwari** — Cloud Engineer & DevOps Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sandeep-tiwari-616a33116/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/sandeeptiwari0206)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-3b82f6?style=flat-square)](https://your-portfolio-url.com)

📍 Jaipur, Rajasthan, India

</div>

---

<div align="center">

⭐ **If this project helped you, give it a star!**

</div>
