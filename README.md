# AutoLance
# 🚀 AutoLance — Freelance Intelligence & Hourly Rate Predictor

**AutoLance** is an end-to-end Data Engineering and Machine Learning project developed as part of the **DEPI Graduation Project**. It aims to analyze the freelancing job market, predict hourly rates for contractors using NLP and Regression models, and provide an interactive Business Intelligence dashboard for real-time decision-making.

---

## 🌟 Key Architecture & Highlights

- **Data Warehousing (Star Schema):** The database is modeled using an optimized Star Schema architecture, separating data into clear Fact (`Fact_Job_Posting`) and Dimension (`Dim_Job`, `Dim_Location`, `Dim_Posting_Attributes`) tables.
- **Machine Learning Pipeline:** A specialized **Ridge Regression** model powered by Natural Language Processing (**TF-IDF**) to extract features from job descriptions and titles, combined with categorical dummy variables.
- **Microservices Architecture:** Complete decoupling between the backend and frontend using a high-performance **FastAPI** REST API.
- **Interactive BI Dashboard:** A custom-built HTML5/CSS3/Chart.js analytics dashboard that dynamically fetches and visualizes real-time market data from the backend.

---

## 🛠️ Tech Stack

| Component | Technologies Used |
| :--- | :--- |
| **Backend & API** | Python, FastAPI, Uvicorn |
| **Machine Learning** | Scikit-Learn (Ridge Regression, TF-IDF), Scipy, NumPy, Pandas |
| **Database Architecture** | MySQL, SQLAlchemy, `mysql-connector-python` |
| **Frontend (ML Interface)** | Streamlit (with custom CSS injection) |
| **Frontend (BI Dashboard)** | HTML5, CSS3, JavaScript, Chart.js |

---

## 📁 Repository Structure

```text
├── app.py                      # FastAPI backend server & database connection endpoints
├── app2.py                     # Streamlit web application for interactive hourly rate prediction
├── Regression model.py         # ML feature engineering, model training, and database extraction script
├── Connection database .py     # Database operations and schema mapping utilities
├── AUTOLANCE DASH BOARD.html   # Custom analytical BI dashboard
├── salary_model.pkl            # Pre-trained ML model artifact (Ridge + TF-IDF vectorizer)
└── requirements.txt            # Project dependencies
```

---


## 📊 Dataset Notice

Due to GitHub's file size limits (the raw dataset is ~170 MB), the primary dataset (`Big_Data_Balanced.csv`) is hosted externally.

- 🔗 **[Download Big_Data_Balanced.csv Here]()**

> **Note:** To run the project from scratch, download the CSV file, ingest it into a local MySQL instance named `Project_Database`, and structure it according to the project's Star Schema design.

---

## 🔒 Security Note

For public security compliance, sensitive database credentials have been replaced with placeholders. Before running the application locally, please update the database password in `app.py` and `Regression model.py`:

```python
password = "YOUR_DATABASE_PASSWORD"  # Replace with your local MySQL password
```

---


🚀 Getting Started & Setup
1. Install Prerequisites
Ensure Python 3.8+ is installed on your machine, then run:

```Bash
pip install -r requirements.txt
```


2. Launch the Backend API (FastAPI)
Start the REST API server to serve database queries:

```Bash
uvicorn app:app --reload
```
The API will run locally at http://127.0.0.1:8000.


3. Open the BI Analytics Dashboard
Once the backend API is running, simply double-click and open AUTOLANCE DASH BOARD.html in any modern web browser (Chrome/Edge) to view interactive charts and job metrics.

4. Launch the AI Prediction Interface (Streamlit)
Open a new terminal window and run:

```Bash
streamlit run app2.py
```
This will open the web interface where users can input job titles and descriptions to get real-time hourly rate predictions.

Developed with ❤️ by the AutoLance Team for DEPI.
