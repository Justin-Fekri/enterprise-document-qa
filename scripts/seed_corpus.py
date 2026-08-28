"""Seed the corpus for every supported chunk size."""

from enterprise_qa.pipeline import get_pipeline
from enterprise_qa.config import CHUNK_SIZE_OPTIONS

if __name__ == "__main__":
    for size in CHUNK_SIZE_OPTIONS:
        pipeline = get_pipeline(size)
        stats = pipeline.stats("admin")
        print(f"chunk_size={size} documents={stats['documents']} chunks={stats['chunks']}")
