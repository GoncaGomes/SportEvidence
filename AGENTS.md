# Repository Guidelines

## Project Overview

SportEvidence is an early-stage Python research assistant for sports-science evidence with the objective of implementig a RAG system to search within a 50 document corpus of science based documentation. Current project exposes CLI commands to:
-  implement single document corpus -> chromaDB storage and  

## Environment
- **Python** >= 3.12
- **Package Manager**: uv (all Python commands must use `uv run`)

## Engineering Principles

- Prefer simple, maintainable solutions.
- Respect existing architectural boundaries.
- Preserve public interfaces unless explicitly instructed otherwise.
- Avoid unnecessary abstractions and dependencies.
- Use explicit types and domain models when useful.
- Keep implementation changes focused.


## Credential Handling

- Never print secret values through reads, searches, shell commands, errors or
  recorded evidence. Choose commands and inspected paths before execution so
  secret-bearing output is never emitted.
- Inspect configuration using variable names and presence or missing status.
  Do not dump environment variables or credential-bearing configuration.
- NEVER read .env
