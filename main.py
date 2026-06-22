import argparse
import sys

from config import CHROMA_DIR, COLLECTION_NAME
from db import get_collection
from ingest import ingest


def cmd_ingest(_: argparse.Namespace) -> None:
    ingest()


def cmd_inspect(_: argparse.Namespace) -> None:
    collection = get_collection()
    count = collection.count()
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Path: {CHROMA_DIR.resolve()}")
    print(f"Records: {count}")

    if count == 0:
        print("\nNo records yet. Add PDFs to documents/ and run: python main.py ingest")
        return

    peek = collection.peek(limit=5)
    print("\nPeek (first 5 records):")
    for i, doc_id in enumerate(peek["ids"]):
        metadata = peek["metadatas"][i] if peek["metadatas"] else {}
        source = metadata.get("source", "?")
        page = metadata.get("page", "?")
        text = peek["documents"][i] if peek["documents"] else ""
        preview = text[:120].replace("\n", " ")
        if len(text) > 120:
            preview += "..."
        print(f"  [{i + 1}] {doc_id}")
        print(f"      source={source}, page={page}")
        print(f"      {preview}")


def cmd_query(args: argparse.Namespace) -> None:
    collection = get_collection()
    if collection.count() == 0:
        print("Collection is empty. Run ingest first.")
        sys.exit(1)

    results = collection.query(
        query_texts=[args.text],
        n_results=args.n_results,
        include=["documents", "metadatas", "distances"],
    )

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print(f'Query: "{args.text}" ({len(ids)} result(s))\n')
    for rank, (doc_id, text, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances), start=1
    ):
        source = metadata.get("source", "?")
        page = metadata.get("page", "?")
        print(f"[{rank}] distance={distance:.4f}  source={source}, page={page}")
        print(f"    id: {doc_id}")
        print(f"    {text}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vector DB setup: ingest PDFs, inspect, and query Chroma."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Load PDFs from documents/ into Chroma")
    ingest_parser.set_defaults(func=cmd_ingest)

    inspect_parser = subparsers.add_parser("inspect", help="Show collection stats and peek records")
    inspect_parser.set_defaults(func=cmd_inspect)

    query_parser = subparsers.add_parser("query", help="Semantic search over ingested chunks")
    query_parser.add_argument("text", help="Search query text")
    query_parser.add_argument(
        "-n",
        "--n-results",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    query_parser.set_defaults(func=cmd_query)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
