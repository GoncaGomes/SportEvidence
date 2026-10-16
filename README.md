# SportEvidence

A research assistant for exploring sports science through evidence-based answers.

SportEvidence is being developed to retrieve, compare, and synthesise findings across scientific documents using retrieval-augmented generation (RAG). The goal is to make research easier to explore while keeping answers traceable to their sources.

## Scope

The project covers running, strength training, athletics, swimming, football, tennis, and other sports, alongside cross-cutting topics such as training, recovery, nutrition, and supplementation.

## Planned Capabilities

- Ingest scientific PDFs and preserve document metadata and source references.
- Search across multiple documents using semantic and keyword retrieval.
- Rerank retrieved passages to improve relevance.
- Generate answers with citations and identify conflicting findings or insufficient evidence.
- Evaluate retrieval quality, citation accuracy, and answer grounding.

## Technical Approach

The planned foundation is a modular Python pipeline with PostgreSQL and pgvector for persistent document, passage, and embedding storage. Retrieval will supply relevant evidence to a language model for synthesis.

Development will start with a measurable RAG baseline. Hierarchical retrieval, graph-based retrieval, and agent workflows will be explored once the core pipeline is validated.

## Initial Corpus

The initial corpus targets 50 credible documents published within the last five years, prioritising systematic reviews, meta-analyses, consensus statements, and relevant primary research. Source selection will record publication details, links, and an inclusion rationale.

## Project Status

Early-stage development. Current priorities are corpus curation, document ingestion, persistent storage, and an end-to-end retrieval baseline.

