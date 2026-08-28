"""CLI wrapper to generate the sample enterprise documents."""

from enterprise_qa.sample_docs import generate_all

if __name__ == "__main__":
    for path in generate_all():
        print(path)
