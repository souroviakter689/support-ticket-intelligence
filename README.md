# AI-Powered Support Ticket Intelligence Platform

An AI-powered support ticket search and intelligence platform that helps users discover relevant customer support tickets using **Semantic Search, TF-IDF, and Hybrid Retrieval**.

🔗 **Live Demo:** [Try the Application](https://support-ticket-intelligence-omoxhwqwjukebnterbjur.streamlit.app/)

---

## 📌 Project Overview

Customer support systems often contain large volumes of tickets describing similar issues in different words. Traditional keyword-based search may fail to identify relevant tickets when users describe the same problem using different expressions.

This project addresses that challenge by combining semantic similarity with traditional TF-IDF-based text retrieval. The system retrieves and ranks support tickets according to their relevance to a user's query.

The platform is designed to demonstrate how Natural Language Processing (NLP), machine learning, and information retrieval techniques can improve support ticket discovery.

## ✨ Key Features

* **Semantic Search:** Uses sentence embeddings to identify tickets with similar meanings.
* **TF-IDF Retrieval:** Measures keyword-based similarity between queries and support tickets.
* **Hybrid Retrieval:** Combines semantic and lexical similarity scores to rank relevant tickets.
* **Ranked Search Results:** Displays the most relevant support tickets for a user query.
* **Similarity Scores:** Provides semantic, TF-IDF, and hybrid relevance scores.
* **Ticket Information:** Displays ticket text, category, and associated cluster information when available.
* **Interactive Web Interface:** Offers an accessible search experience through Streamlit.
* **Live Deployment:** Accessible through Streamlit Community Cloud.

## 🧠 How It Works

The platform follows a hybrid information retrieval pipeline:

1. **Query Input:** The user enters a customer support question or issue.
2. **Semantic Representation:** The query is converted into a numerical embedding using Sentence Transformers.
3. **Semantic Similarity:** The query embedding is compared with stored ticket embeddings using cosine similarity.
4. **Keyword Similarity:** TF-IDF features are used to calculate lexical similarity.
5. **Hybrid Ranking:** Both similarity scores are combined using a weighted scoring formula.
6. **Result Display:** The application displays the top-ranked tickets and their relevance scores.

### Hybrid Retrieval Formula

The hybrid score is calculated as:

**Hybrid Score = α × Semantic Similarity + (1 − α) × TF-IDF Similarity**

The current configuration uses:

* Semantic similarity weight (α): **0.9**
* TF-IDF similarity weight: **0.1**

This configuration gives greater importance to semantic similarity while retaining keyword-based matching.

## 🛠️ Technology Stack

| Component                  | Technology                               |
| -------------------------- | ---------------------------------------- |
| Programming Language       | Python                                   |
| Web Interface              | Streamlit                                |
| NLP Embeddings             | Sentence Transformers                    |
| Embedding Model            | `sentence-transformers/all-MiniLM-L6-v2` |
| Text Representation        | TF-IDF                                   |
| Similarity Measurement     | Cosine Similarity                        |
| Machine Learning           | Scikit-learn                             |
| Data Processing            | NumPy, Pandas, SciPy                     |
| Model and Artifact Storage | Joblib, NumPy                            |
| Deployment                 | Streamlit Community Cloud                |
| Version Control            | Git and GitHub                           |

## 🏗️ System Architecture

The application follows this retrieval workflow:

**User Query → Sentence Embedding → Semantic Similarity + TF-IDF Similarity → Hybrid Ranking → Ranked Support Tickets**

Precomputed embeddings, the TF-IDF vectorizer, and supporting artifacts are used to perform retrieval without rebuilding the entire dataset representation for every query.

## 📂 Project Structure

```text
support-ticket-intelligence/
│
├── artifacts/
│   ├── cluster_assignments.csv
│   ├── config.json
│   ├── kmeans_model.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── train_corpus.csv
│   ├── train_embeddings.npy
│   └── train_tfidf.npz
│
├── app.py
├── main.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── test_api.py
├── .dockerignore
└── README.md
```

## ⚙️ Model Configuration

| Parameter                   | Value                                    |
| --------------------------- | ---------------------------------------- |
| Embedding Model             | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding Dimension         | 384                                      |
| Retrieval Strategy          | Hybrid                                   |
| Semantic Weight             | 0.9                                      |
| TF-IDF Weight               | 0.1                                      |
| Default Top-K Results       | 5                                        |
| Maximum Search Results      | 20                                       |
| Configured K-Means Clusters | 77                                       |
| Random State                | 42                                       |

These values represent the current project configuration.

## 🚀 Live Application

Try the deployed application:

**[Launch Support Ticket Intelligence](https://support-ticket-intelligence-omoxhwqwjukebnterbjur.streamlit.app/)**

Enter a support-related question to retrieve and explore relevant tickets.

### Example Queries

* I want to know my account balance.
* My cash deposit has not been reflected in my account.
* I am having trouble with a recent transaction.
* I need help with my account.

The retrieved results depend on the available dataset and the user's query.

## 💻 Running Locally

To run the Streamlit application locally:

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies listed in `requirements.txt`.
4. Start the Streamlit application.

Refer to the project files for the relevant setup and execution details.

## 🔌 API Component

The repository also includes a FastAPI implementation in `main.py`, with endpoints intended for health checks and support ticket search.

* `GET /health` — checks the API status.
* `POST /search` — retrieves relevant support tickets.

The deployed Streamlit application performs retrieval directly within the Streamlit app. The FastAPI implementation is retained for local testing and potential future API-based deployment.

## 🔮 Future Improvements

Potential extensions include:

* Advanced query understanding and intent detection.
* Category-based filtering.
* Improved retrieval evaluation using Precision@K, Recall@K, and Mean Reciprocal Rank.
* Comparison of alternative embedding models.
* Feedback-based relevance improvement.
* Integration with customer support platforms.
* A dedicated backend API deployment.

## 👩‍💻 Author

**Sourovi Akter**

Interested in Artificial Intelligence, Machine Learning, Natural Language Processing, and Data Science.

🔗 **GitHub:** [souroviakter689](https://github.com/souroviakter689)

---

⭐ If you find this project interesting, consider starring the repository!
