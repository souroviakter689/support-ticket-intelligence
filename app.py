import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from scipy.sparse import load_npz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Support Ticket Intelligence",
    page_icon="🎫",
    layout="wide"
)

torch.set_num_threads(2)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"


# --------------------------------------------------
# 2. Load model and artifacts once
# --------------------------------------------------

@st.cache_resource
def load_resources():

    with open(ARTIFACTS_DIR / "config.json", "r") as f:
        config = json.load(f)

    embedding_model = SentenceTransformer(
        config["embedding_model"],
        device="cpu"
    )

    tfidf_vectorizer = joblib.load(
        ARTIFACTS_DIR / "tfidf_vectorizer.joblib"
    )

    train_embeddings = np.load(
        ARTIFACTS_DIR / "train_embeddings.npy",
        mmap_mode="r"
    )

    train_tfidf = load_npz(
        ARTIFACTS_DIR / "train_tfidf.npz"
    )

    train_corpus = pd.read_csv(
        ARTIFACTS_DIR / "train_corpus.csv"
    )

    cluster_assignments = pd.read_csv(
        ARTIFACTS_DIR / "cluster_assignments.csv"
    )

    alpha = float(config["selected_alpha"])

    return (
        config,
        embedding_model,
        tfidf_vectorizer,
        train_embeddings,
        train_tfidf,
        train_corpus,
        cluster_assignments,
        alpha
    )


# --------------------------------------------------
# 3. Search function
# --------------------------------------------------

def search_tickets(query, top_k, resources):

    (
        config,
        embedding_model,
        tfidf_vectorizer,
        train_embeddings,
        train_tfidf,
        train_corpus,
        cluster_assignments,
        alpha
    ) = resources

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    semantic_scores = cosine_similarity(
        query_embedding,
        train_embeddings
    )[0]

    query_tfidf = tfidf_vectorizer.transform([query])

    tfidf_scores = cosine_similarity(
        query_tfidf,
        train_tfidf
    )[0]

    hybrid_scores = (
        alpha * semantic_scores
        + (1 - alpha) * tfidf_scores
    )

    top_indices = np.argsort(
        hybrid_scores
    )[::-1][:top_k]

    results = []

    for idx in top_indices:

        results.append({
            "matched_query": str(
                train_corpus.iloc[idx]["text"]
            ),
            "category": str(
                train_corpus.iloc[idx]["category"]
            ),
            "semantic_similarity": round(
                float(semantic_scores[idx]), 4
            ),
            "tfidf_similarity": round(
                float(tfidf_scores[idx]), 4
            ),
            "hybrid_score": round(
                float(hybrid_scores[idx]), 4
            ),
            "cluster": int(
                cluster_assignments.iloc[idx]["cluster"]
            )
        })

    return results, alpha


# --------------------------------------------------
# 4. Page header
# --------------------------------------------------

st.title("🎫 AI-Powered Support Ticket Intelligence")

st.markdown(
    """
    Search similar customer support tickets using
    **Hybrid Semantic Search**.

    Enter a customer issue below to discover related tickets,
    categories, similarity scores, and clusters.
    """
)

st.divider()


# --------------------------------------------------
# 5. Load resources
# --------------------------------------------------

try:

    with st.spinner("Loading AI model and ticket data..."):
        resources = load_resources()

except Exception as exc:

    st.error("Unable to load the model or saved artifacts.")
    st.exception(exc)
    st.stop()


# --------------------------------------------------
# 6. Search input
# --------------------------------------------------

query = st.text_area(
    "Customer Support Query",
    placeholder="Example: My card payment did not go through",
    height=100
)

top_k = st.slider(
    "Number of results",
    min_value=1,
    max_value=20,
    value=5
)


# --------------------------------------------------
# 7. Search button
# --------------------------------------------------

if st.button("🔍 Search Tickets", type="primary"):

    if not query.strip():

        st.warning("Please enter a customer support query.")

    else:

        try:

            with st.spinner("Searching for similar support tickets..."):

                results, alpha = search_tickets(
                    query.strip(),
                    top_k,
                    resources
                )

            st.success(f"Found {len(results)} matching tickets.")

            col1, col2 = st.columns(2)

            col1.metric("Results Found", len(results))

            col2.metric(
                "Semantic Search Weight",
                f"{alpha:.0%}"
            )

            st.divider()

            for i, result in enumerate(results, start=1):

                with st.container(border=True):

                    st.subheader(
                        f"Result {i}: {result['category']}"
                    )

                    st.write(result["matched_query"])

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Hybrid Score",
                        f"{result['hybrid_score']:.4f}"
                    )

                    col2.metric(
                        "Semantic Similarity",
                        f"{result['semantic_similarity']:.4f}"
                    )

                    col3.metric(
                        "TF-IDF Similarity",
                        f"{result['tfidf_similarity']:.4f}"
                    )

                    st.caption(
                        f"K-Means Cluster: {result['cluster']}"
                    )

        except Exception as exc:

            st.error("Search failed.")
            st.exception(exc)


# --------------------------------------------------
# 8. Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Powered by Sentence Transformers, TF-IDF, "
    "Hybrid Retrieval, and Streamlit."
)
