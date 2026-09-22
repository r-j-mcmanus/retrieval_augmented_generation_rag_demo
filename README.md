# Retrieval Augmented Generation (RAG) Project

This project is a toy RAG pipeline for indexing and querying different document types using chunking, embeddings, vector search, and a local LLM.

It currently supports:
- PDF files
- VTT subtitle files
- MP3 audio transcripts
- HTML pages

The design is modular via use of the interface pattern allowing additional source types implementing an additional extractor and registering it with the pipeline.

## Set Up

run the file `run_all.bat`

## Architecture

The repository is organized around a few core pieces:

- Extractors: convert source files into text chunks and metadata
- Embedding layer: turns chunk text into vector embeddings
- Storage layer: stores chunk metadata and vector representations
- LLM layer: generates answers from retrieved context
- Pipeline: coordinates indexing and retrieval

## Relational metadata model

The SQLite metadata store uses these normalized tables:

- `documents`: document title, content, type, visibility, timestamps, and metadata
- `document_clients`: many-to-many document/client references
- `tags`: unique internal-document tag names
- `document_tags`: many-to-many document/tag links
- `chunk_records`: chunk content and the `doc_id` link back to `documents`

`index_file(..., client_reference=...)` writes a client relationship, while
`index_file(..., tags=[...])` associates tags with an internal document. Client
scope filters use `document_clients`, so documents cannot be selected for a
client unless that relationship exists. Existing SQLite databases are migrated
from the former `file_metadata` table when the store starts.

## Document Extractors

The project includes the following extractor implementations:

- PDFExtractor
- VTTExtractor
- MP3Extractor
- HTMLExtractor

Each extractor follows the same base contract:
- extract(file_path) -> returns a list of chunk objects
- get_useful_metadata(file_path) -> returns metadata for the source

## Pipeline flow

1. A file is passed to the RAG pipeline.
2. The correct extractor is selected by file extension.
3. The extractor turns the file into text chunks.
4. Each chunk is embedded with the configured encoder.
5. The vectors and metadata are saved in the chosen storage backends.
6. A user query is embedded and compared against stored vectors.
7. The most relevant chunks are retrieved and sent to the LLM as context.
8. The LLM responds using only the retrieved source content.

## Adding a new extractor

Create a new extractor class that inherits from BaseDocumentExtractor.

Then expose it in [extractors/__init__.py](extractors/__init__.py) and add it to the extractor list used by the pipeline.
