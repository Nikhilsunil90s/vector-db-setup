import streamlit as st

from config import CHROMA_DIR, COLLECTION_NAME, DOCUMENTS_DIR
from db import get_collection
from ingest import ingest


@st.cache_resource
def load_collection():
    return get_collection()


def main() -> None:
    st.set_page_config(page_title="Vector DB Viewer", layout="wide")
    st.title("Vector DB Viewer")

    collection = load_collection()
    count = collection.count()

    with st.sidebar:
        st.header("Database")
        st.text(f"Path: {CHROMA_DIR.resolve()}")
        st.text(f"Collection: {COLLECTION_NAME}")
        st.metric("Documents", count)

        if st.button("Re-ingest PDFs"):
            with st.spinner("Ingesting PDFs..."):
                ingest()
            load_collection.clear()
            st.rerun()

        st.divider()
        st.header("Search")
        query = st.text_input("Query", placeholder="Enter search text...")
        n_results = st.slider("Results", min_value=1, max_value=20, value=5)

    if count == 0:
        st.info(
            f"No documents in the collection. Add PDFs to `{DOCUMENTS_DIR.resolve()}/` "
            "and click **Re-ingest PDFs** in the sidebar, or run `uv run python main.py ingest`."
        )
        return

    tab_search, tab_peek = st.tabs(["Search", "Peek"])

    with tab_search:
        if query.strip():
            results = collection.query(
                query_texts=[query.strip()],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
            ids = results["ids"][0]
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            st.subheader(f'Results for "{query.strip()}"')
            for rank, (doc_id, text, metadata, distance) in enumerate(
                zip(ids, documents, metadatas, distances), start=1
            ):
                source = metadata.get("source", "?")
                page = metadata.get("page", "?")
                with st.expander(
                    f"#{rank} — {source} (page {page}) · distance {distance:.4f}",
                    expanded=rank == 1,
                ):
                    st.caption(f"ID: {doc_id}")
                    st.write(text)
        else:
            st.caption("Enter a query in the sidebar to search ingested chunks.")

    with tab_peek:
        peek_limit = st.number_input("Records to show", min_value=1, max_value=100, value=10)
        peek = collection.get(
            limit=peek_limit,
            include=["documents", "metadatas"],
        )
        if peek["ids"]:
            rows = []
            for i, doc_id in enumerate(peek["ids"]):
                metadata = peek["metadatas"][i] if peek["metadatas"] else {}
                text = peek["documents"][i] if peek["documents"] else ""
                rows.append(
                    {
                        "id": doc_id,
                        "source": metadata.get("source", ""),
                        "page": metadata.get("page", ""),
                        "text": text[:200] + ("..." if len(text) > 200 else ""),
                    }
                )
            st.dataframe(rows, width=True)
        else:
            st.caption("No records to display.")


if __name__ == "__main__":
    main()
