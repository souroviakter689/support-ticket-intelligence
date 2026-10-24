
import os
import requests
import streamlit as st


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Support Ticket Intelligence",
    page_icon="🎫",
    layout="wide"
)


# --------------------------------------------------
# 2. API configuration
# --------------------------------------------------

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


# --------------------------------------------------
# 3. Page header
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
# 4. Search input
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
# 5. Search button and API request
# --------------------------------------------------

if st.button("🔍 Search Tickets", type="primary"):

    if not query.strip():
        st.warning("Please enter a customer support query.")

    else:
        payload = {
            "query": query,
            "top_k": top_k
        }

        try:
            with st.spinner("Searching for similar support tickets..."):

                response = requests.post(
                    f"{API_URL}/search",
                    json=payload,
                    timeout=60
                )

            if response.status_code == 200:

                data = response.json()

                st.success(
                    f"Found {data['result_count']} matching tickets."
                )

                col1, col2 = st.columns(2)

                col1.metric(
                    "Results Found",
                    data["result_count"]
                )

                col2.metric(
                    "Semantic Search Weight",
                    f"{data['selected_alpha']:.0%}"
                )

                st.divider()

                for i, result in enumerate(
                    data["results"],
                    start=1
                ):

                    with st.container(border=True):

                        st.subheader(
                            f"Result {i}: {result['category']}"
                        )

                        st.write(
                            result["matched_query"]
                        )

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

            else:
                st.error(
                    f"API Error ({response.status_code}): "
                    f"{response.text}"
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to the FastAPI server. "
                "Please make sure the backend is running."
            )

        except requests.exceptions.Timeout:
            st.error(
                "The search request timed out. "
                "Please try again."
            )

        except requests.exceptions.RequestException as exc:
            st.error(
                f"Unable to complete the request: {exc}"
            )


# --------------------------------------------------
# 6. Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Powered by Sentence Transformers, TF-IDF, "
    "Hybrid Retrieval, FastAPI, and Streamlit."
)
