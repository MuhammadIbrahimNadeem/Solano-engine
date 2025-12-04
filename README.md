# Solano Engine - Advanced Information Retrieval System

**Course:** CS 516: Information Retrieval and Text Mining  
**Instructor:** Dr. Ahmad Mustafa  
**Fall 2025**

## Overview

Solano Engine is a local, lightweight Information Retrieval (IR) system designed to index and search through news articles efficiently. It implements a **Vector Space Model** using **TF-IDF** weighting and **Cosine Similarity** for ranking. The system features a modern, responsive web interface inspired by premium search engines.

### Key Features
*   **Fast Indexing:** Uses sparse matrices for memory-efficient storage.
*   **Dual Search Modes:**
    *   **Regular Search:** Natural language queries ranked by relevance.
    *   **Boolean Search:** Supports `AND`, `OR`, `NOT` operators (e.g., `python AND coding NOT java`).
*   **Real-time Evaluation:** Calculates Precision@10, Recall, F1-Score, MAP, and NDCG for every query.
*   **Smart UI:**
    *   "Read Time" estimates.
    *   Confidence score badges.
    *   Sticky metrics sidebar.
    *   Dark mode with glassmorphism effects.
*   **Related Articles:** Recommendation system to find similar news.

---

## 🛠️ Prerequisites

*   **Python 3.8+**
*   **pip** (Python package manager)

---

## 🚀 Installation & Setup

1.  **Clone the Repository** (or unzip the project folder):
    ```bash
    cd ir_news
    ```

2.  **Create a Virtual Environment (Optional but Recommended):**
    ```bash
    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Verify Data:**
    Ensure the `data.csv` file is present in the root directory. This file contains the news articles to be indexed.

---

## ▶️ How to Run

1.  **Start the Application:**
    ```bash
    python app.py
    ```

2.  **Access the Search Engine:**
    Open your web browser and navigate to:
    👉 **http://localhost:5001**

---

## 📂 Project Structure

```
Solano-Engine/
├── app.py              # Main Flask application & Data Loader
├── helpers.py          # Search logic, Boolean parsing, & Metrics
├── data.csv            # Dataset (News Articles)
├── requirements.txt    # Python dependencies
├── static/
│   └── style.css       # CSS Styling (Dark mode, Glassmorphism)
└── templates/
    ├── index.html      # Home Page (Search Bar)
    └── results.html    # Results Page (Listings + Metrics)
```

## 🔍 Usage Examples

*   **Basic Search:** `stock market`
*   **Boolean AND:** `apple AND samsung`
*   **Boolean OR:** `football OR soccer`
*   **Boolean NOT:** `apple NOT fruit`
*   **Complex:** `technology AND (AI OR robotics)`

