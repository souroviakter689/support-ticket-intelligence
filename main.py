
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from scipy.sparse import load_npz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. Configure logging
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# 2. Load saved artifacts
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

with open(ARTIFACTS_DIR / "config.json", "r") as f:
    config = json.load(f)

embedding_model = SentenceTransformer(
    config["embedding_model"]
)

tfidf_vectorizer = joblib.load(
    ARTIFACTS_DIR / "tfidf_vectorizer.joblib"
)

train_embeddings = np.load(
    ARTIFACTS_DIR / "train_embeddings.npy"
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

logger.info("Saved artifacts loaded successfully.")


# --------------------------------------------------
# 3. Initialize FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Support Ticket Intelligence API",
    description="Hybrid semantic search for support queries.",
    version="1.0.0"
)


# --------------------------------------------------
# 4. Define request schema
# --------------------------------------------------

class SearchRequest(BaseModel):

    query: str = Field(
        min_length=1,
        max_length=1000,
        description="The support query to search for."
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of matching results to return."
    )


# --------------------------------------------------
# 5. Health endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "retrieval_method": config["retrieval_method"],
        "embedding_model": config["embedding_model"]
    }


# --------------------------------------------------
# 6. Search endpoint
# --------------------------------------------------

@app.post("/search")
def search(request: SearchRequest):

    query = request.query.strip()

    # Reject empty or whitespace-only queries
    if not query:
        raise HTTPException(
            status_code=422,
            detail="Query cannot be empty or whitespace only."
        )

    try:

        # Generate query embedding
        query_embedding = embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Calculate semantic similarity
        semantic_scores = cosine_similarity(
            query_embedding,
            train_embeddings
        )[0]

        # Calculate TF-IDF similarity
        query_tfidf = tfidf_vectorizer.transform(
            [query]
        )

        tfidf_scores = cosine_similarity(
            query_tfidf,
            train_tfidf
        )[0]

        # Calculate hybrid scores
        hybrid_scores = (
            alpha * semantic_scores
            + (1 - alpha) * tfidf_scores
        )

        # Select top-k results
        top_indices = np.argsort(
            hybrid_scores
        )[::-1][:request.top_k]

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

        logger.info(
            "Search completed successfully. Results: %d",
            len(results)
        )

        return {
            "query": query,
            "selected_alpha": alpha,
            "result_count": len(results),
            "results": results
        }

    except HTTPException:
        # Preserve intentional HTTP errors
        raise

    except Exception:
        # Record the full error in server logs
        logger.exception(
            "Unexpected error during support ticket search."
        )

        # Return a safe message to the API client
        raise HTTPException(
            status_code=500,
            detail=(
                "An internal error occurred "
                "while processing the search."
            )
        )
