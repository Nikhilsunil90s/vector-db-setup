from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, DOCUMENTS_DIR
from db import get_collection


def _chunk_id(source: str, page: int, index: int) -> str:
    return f"{source}::page_{page}::chunk_{index}"


def ingest() -> None:
    pdf_paths = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {DOCUMENTS_DIR.resolve()}/")
        print("Add PDF files to that folder and run ingest again.")
        return

    collection = get_collection()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    total_chunks = 0
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        chunks = splitter.split_documents(documents)

        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict] = []

        for index, chunk in enumerate(chunks):
            page = chunk.metadata.get("page", 0)
            source = pdf_path.name
            ids.append(_chunk_id(source, page, index))
            texts.append(chunk.page_content)
            metadatas.append({"source": source, "page": page})

        collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
        total_chunks += len(chunks)
        print(f"  {pdf_path.name}: {len(chunks)} chunks")

    print(f"\nProcessed {len(pdf_paths)} file(s), upserted {total_chunks} chunk(s).")
    print(f"Collection total: {collection.count()} record(s).")


if __name__ == "__main__":
    ingest()
