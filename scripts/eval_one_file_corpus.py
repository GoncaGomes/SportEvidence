import json
import time
from datetime import datetime
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

import chromadb
from chromadb.api.client import Client

from sportevidence.ingestion.embeddings import embed_texts
from sportevidence.models import SearchHit
from sportevidence.storage.vector_store import search_chunks

EVAL_CASES = [
    {
        "id": "Q01",
        "question": "How does the review define running economy?",
        "expression": "running economy definition",
        "expected_text": [
            "Running economy is defined as the energy demand at submaximal running speed",
        ],
    },
    {
        "id": "Q02",
        "question": (
            "What were the minimum duration and weekly frequency "
            "for a strength training programme to qualify for inclusion?"
        ),
        "expression": "strength training intervention eligibility minimum duration weekly sessions",
        "expected_text": [
            "lasting ≥ 3 weeks, with ≥ 1 weekly session",
        ],
    },
    {
        "id": "Q03",
        "question": (
            "In the plyometric training subgroup analysis, how did the effect "
            "on running economy differ between speeds at or below 12 km/h "
            "and speeds above 12 km/h?"
        ),
        "expression": "plyometric training running economy subgroup effect sizes ≤12 km/h >12 km/h",
        "expected_text": [
            "≤ 12.00 km/h | 10 | 18 | - 0.307",
            "t(25) = - 2.802, p = 0.010",
            "> 12.00 km/h | 5 | 9 | 0.163",
            "t(25) = 1.104, p = 0.280",
        ],
    },
    {
        "id": "Q04",
        "question": (
            "Which two factors were associated with larger improvements "
            "in running economy after high-load strength training?"
        ),
        "expression": "high-load strength training running economy effect moderators",
        "expected_text": [
            "modulated the effect of high load strength training on running economy "
            "(i.e., greater improvements at higher speeds and higher VO 2 max)",
        ],
    },
    {
        "id": "Q05",
        "question": (
            "In Turner et al., what were the mean running economy values "
            "before and after plyometric training at 9.65 km/h?"
        ),
        "expression": "Turner et al. plyometric training running economy 9.65 km/h pre post",
        "expected_text": [
            "Turner et al. [106] | PL",
            "Turner et al. [106] |  |  | m/ml/kg | 9.65 km/h | "
            "< 11.50 km/h |  |  | 5.14 (0.36) | 5.30 (0.36)",
        ],
    },
]


def retrieve_passages(
    collection_name: str = "sports",
    db_path: Path = Path("tests/fixtures/data"),
    *,
    question: str,
) -> list[SearchHit]:
    persistent_client = cast(Client, chromadb.PersistentClient(path=str(db_path)))
    with persistent_client as client:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=None,
        )
        question_to_make = [question]
        list_embeddings = embed_texts(question_to_make)
        hit = search_chunks(collection, list_embeddings[0], limit=3)
    return hit


def main() -> int:
    results = []

    for case in EVAL_CASES:
        begining = time.monotonic()
        hits = retrieve_passages(question=case["question"])
        end = time.monotonic()
        duration = end - begining
        results.append(
            {
                **case,
                "hits": [
                    {
                        "rank": rank,
                        "chunk_id": hit.chunk_id,
                        "page_numbers": list(hit.page_numbers),
                        "distance": hit.distance,
                        "text": hit.text,
                    }
                    for rank, hit in enumerate(hits, start=1)
                ],
                "elapsed time in seconds": duration,
            }
        )
    timestamp = datetime.now(ZoneInfo("Europe/Lisbon")).strftime("%Y%m%dT%H%M%S%f")
    output_path = Path("tests/reports") / f"evaluation_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps({"cases": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
