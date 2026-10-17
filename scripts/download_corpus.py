"""Download and verify the corpus without changing its bibliographic catalogue.

Run with the project's Python environment. Existing successful downloads are
rechecked and reused; only missing/failed entries are downloaded again.
--verify-only checks the catalogue, every payload, and its recorded SHA-256 offline.

PMC API: https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/
Dataset schema: https://pmc-oa-opendata.s3.amazonaws.com/README.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlsplit
from zipfile import BadZipFile, ZipFile

import requests
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text
from pdfminer.pdfpage import PDFPage
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = ROOT / "tests/corpus_desporto_50.json"
OUTPUT = ROOT / "tests/fixtures"
REPORT = OUTPUT / "relatorio_downloads.json"
S3 = "https://pmc-oa-opendata.s3.amazonaws.com/"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
ALTERNATIVES = {
    "SPORT-032": [
        "https://bjsm.bmj.com/content/bjsports/57/17/1073.full.pdf",
        (
            "https://pure.amsterdamumc.nl/files/136801390/"
            "2023-international-olympic-committees-ioc-consensus-statement-on-"
            "relative-energy-deficiency-in-sp.pdf"
        ),
        (
            "https://www.nationaleatingdisorders.org/wp-content/uploads/2024/01/"
            "RED.S-Consensus.-September-2023.-British-Journal-of-Medicine.pdf"
        ),
    ],
    "SPORT-045": [
        (
            "https://www.storre.stir.ac.uk/retrieve/aed7585f-24dc-48cc-a927-4914b469df27/"
            "Manuscript%20-%20July%201%202024.docx"
        ),
    ],
}
logging.getLogger("pdfminer").setLevel(logging.ERROR)


def now():
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def compact(value):
    value = unicodedata.normalize("NFKD", value).casefold()
    return "".join(c for c in value if c.isalnum())


def tokens(value):
    value = unicodedata.normalize("NFKD", value).casefold()
    return set(re.findall(r"[a-z0-9]+", value))


def title_check(expected, observed):
    expected_tokens = tokens(expected)
    coverage = len(expected_tokens & tokens(observed)) / len(expected_tokens)
    exact = compact(expected) in compact(observed)
    return exact, round(coverage, 4)


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(data)
    temporary.replace(path)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


class Download:
    def __init__(self, doc):
        self.doc = doc
        self.attempts = []
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "SportEvidence/0.1 (corpus acquisition)"
        retry = Retry(
            total=2, backoff_factor=0.75, status_forcelist=[429, 500, 502, 503, 504]
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def get(self, url, purpose, **kwargs):
        attempt = {"url": url, "finalidade": purpose, "inicio_utc": now()}
        self.attempts.append(attempt)
        try:
            response = self.session.get(url, timeout=(15, 60), **kwargs)
            attempt.update(
                {
                    "url_pedido": response.request.url,
                    "url_final": response.url,
                    "http_status": response.status_code,
                    "content_type": response.headers.get("Content-Type"),
                    "bytes": len(response.content),
                    "sha256_resposta": sha256(response.content),
                    "content_disposition": response.headers.get("Content-Disposition"),
                    "redirecionamentos": [r.url for r in response.history],
                    "repeticoes_http": len(response.raw.retries.history),
                }
            )
            response.raise_for_status()
            return response, attempt
        except requests.RequestException as exc:
            attempt["erro"] = str(exc)
            raise

    def save(self, response, attempt, metadata=None):
        data = response.content
        fmt, evidence = inspect_payload(data, self.doc, metadata)
        # S3 publishes MD5 per object; also calculate our own independent SHA-256.
        expected_md5 = parse_qs(urlsplit(response.url).query).get("md5", [None])[0]
        if expected_md5:
            actual_md5 = hashlib.md5(data, usedforsecurity=False).hexdigest()
            if actual_md5 != expected_md5:
                raise ValueError("MD5 differs from the official dataset manifest")
            evidence["md5_fonte_confere"] = True
        path = OUTPUT / f"{self.doc['id']}.{fmt}"
        atomic_write(path, data)
        attempt["aceite"] = True
        return {
            "estado": "sucesso",
            "ficheiro": path.relative_to(ROOT).as_posix(),
            "formato": fmt,
            "content_type": response.headers.get("Content-Type"),
            "bytes": len(data),
            "sha256": sha256(data),
            "url_download": response.request.url,
            "url_final": response.url,
            "obtido_em_utc": now(),
            "verificacao": evidence,
        }

    def pmc(self):
        pmcid = self.doc["pmcid"]
        response, _ = self.get(
            S3,
            "listar versões oficiais PMC",
            params={
                "list-type": "2",
                "prefix": pmcid + ".",
                "delimiter": "/",
            },
        )
        tree = ET.fromstring(response.content)
        prefixes = [n.text for n in tree.findall("s3:CommonPrefixes/s3:Prefix", NS)]
        if tree.findtext("s3:IsTruncated", namespaces=NS) == "true":
            raise ValueError(
                "Truncated version listing; refusing an incomplete selection"
            )
        candidates = []
        for prefix in prefixes:
            version = prefix.rstrip("/")
            response, attempt = self.get(
                S3 + prefix + version + ".json", "metadados PMC"
            )
            meta = response.json()
            exact, coverage = title_check(self.doc["titulo"], meta.get("title", ""))
            if (
                meta.get("pmcid") != pmcid
                or compact(meta.get("doi") or "") != compact(self.doc["doi"])
                or not (exact or coverage >= 0.95)
            ):
                attempt["erro_validacao"] = "PMCID, DOI or title differs from catalogue"
                continue
            candidates.append((meta, response, attempt))
        # A higher version may be a manuscript: prefer the published version.
        candidates.sort(
            key=lambda c: (
                bool(c[0].get("is_manuscript")),
                -int(c[0].get("version", 0)),
            )
        )
        for meta, meta_response, meta_attempt in candidates:
            for key in ("pdf_url", "xml_url"):
                source = meta.get(key)
                if not source:
                    continue
                if source.startswith("s3://pmc-oa-opendata/"):
                    source = S3 + source.removeprefix("s3://pmc-oa-opendata/")
                try:
                    response, attempt = self.get(source, "documento integral PMC")
                    result = self.save(response, attempt, meta)
                    meta_path = OUTPUT / "metadata" / f"{self.doc['id']}.pmc.json"
                    atomic_write(meta_path, meta_response.content)
                    result["metadados_fonte"] = {
                        "ficheiro": meta_path.relative_to(ROOT).as_posix(),
                        "sha256": sha256(meta_response.content),
                        "url": meta_attempt["url_final"],
                    }
                    return result
                except (requests.RequestException, ValueError, ET.ParseError) as exc:
                    self.attempts[-1]["erro_validacao"] = str(exc)
        # Europe PMC provides open access full text as JATS XML.
        source = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
        response, attempt = self.get(source, "alternativa XML integral Europe PMC")
        return self.save(response, attempt)

    def external(self):
        sources = [
            self.doc["link_texto_integral"],
            *ALTERNATIVES.get(self.doc["id"], []),
        ]
        visited = set()
        while sources:
            url = sources.pop(0)
            if url in visited:
                continue
            visited.add(url)
            try:
                response, attempt = self.get(url, "documento ou página institucional")
                if response.content.startswith((b"%PDF-", b"PK\x03\x04")):
                    return self.save(response, attempt)
                soup = BeautifulSoup(response.content, "html.parser")
                # Landing pages are never accepted as the full document.
                attempt["erro_validacao"] = (
                    "Página HTML; procurar ligação ao documento integral"
                )
                for a in soup.find_all("a", href=True):
                    href = urljoin(response.url, a["href"])
                    label = a.get_text(" ", strip=True)
                    exact, coverage = title_check(self.doc["titulo"], label)
                    if ".pdf" in urlsplit(href).path.lower() and (
                        exact or coverage >= 0.8
                    ):
                        sources.append(href)
                for tag in soup.find_all("meta", attrs={"name": "citation_pdf_url"}):
                    sources.append(urljoin(response.url, tag.get("content", "")))
            except (
                requests.RequestException,
                ValueError,
                ET.ParseError,
                BadZipFile,
            ) as exc:
                self.attempts[-1]["erro_validacao"] = str(exc)
        raise ValueError(
            "Nenhuma fonte devolveu um documento integral com identidade confirmada"
        )

    def run(self):
        result = {
            "id": self.doc["id"],
            "titulo": self.doc["titulo"],
            "doi": self.doc.get("doi"),
            "pmcid": self.doc.get("pmcid"),
            "url_catalogo": self.doc["link_texto_integral"],
        }
        try:
            result.update(self.pmc() if self.doc.get("pmcid") else self.external())
        except Exception as exc:  # noqa: BLE001 - retain a failure record for every catalogue ID
            if self.attempts:
                self.attempts[-1].setdefault("erro_validacao", str(exc))
            result.update(
                {
                    "estado": "falha",
                    "erro": f"{type(exc).__name__}: {exc}",
                    "ficheiro": None,
                    "formato": None,
                    "sha256": None,
                }
            )
        finally:
            self.session.close()
        result["tentativas"] = self.attempts
        return result


def inspect_payload(data, doc, metadata=None):
    evidence = {"identidade_confirmada": False}
    if data.startswith(b"%PDF-"):
        fmt = "pdf"
        if b"%%EOF" not in data[-2048:]:
            raise ValueError("PDF missing end-of-file marker")
        page_count = sum(1 for _ in PDFPage.get_pages(BytesIO(data)))
        text = extract_text(BytesIO(data), maxpages=2)
        if page_count < 2 or len(text.strip()) < 1500:
            raise ValueError("PDF has insufficient article text or pages")
        evidence.update(
            {"paginas": page_count, "paginas_texto_verificado": min(2, page_count)}
        )
    elif data.startswith(b"PK\x03\x04"):
        fmt = "docx"
        with ZipFile(BytesIO(data)) as archive:
            bad_member = archive.testzip()
            if bad_member:
                raise ValueError(f"Corrupt DOCX member: {bad_member}")
            tree = ET.fromstring(archive.read("word/document.xml"))
            paragraphs = tree.findall(
                ".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"
            )
            text = "\n".join("".join(p.itertext()) for p in paragraphs)
        if len(text.strip()) < 10000:
            raise ValueError("DOCX has insufficient full text")
        evidence["carateres_texto_integral"] = len(text)
        if "accepted manuscript" in text[:1000].lower():
            evidence["versao_documento"] = "manuscrito aceite"
    else:
        fmt = "xml"
        tree = ET.fromstring(data)
        if tree.tag != "article" or tree.get("article-type") in {
            "correction",
            "retraction",
        }:
            raise ValueError("Not a full article JATS XML")
        front = tree.find("front/article-meta")
        body = tree.find("body")
        if front is None or body is None or len("".join(body.itertext())) < 5000:
            raise ValueError("XML contains no substantive full article body")
        ids = {
            n.get("pub-id-type"): "".join(n.itertext())
            for n in front.findall("article-id")
        }
        if compact(ids.get("doi", "")) != compact(doc.get("doi", "")):
            raise ValueError("JATS article DOI differs from catalogue")
        if (
            doc.get("pmcid")
            and ids.get("pmcid", "PMC" + ids.get("pmc", "")) != doc["pmcid"]
        ):
            raise ValueError("JATS article PMCID differs from catalogue")
        text = "".join(front.find("title-group/article-title").itertext())
        text += "\n" + ids.get("doi", "")
        evidence["identificadores_xml"] = ids
    # Use the title/front matter, never references, to identify the article.
    identity_text = text[:18000] if fmt == "docx" else text
    exact, coverage = title_check(doc["titulo"], identity_text)
    doi = doc.get("doi") or ""
    doi_present = bool(doi and compact(doi) in compact(identity_text))
    if not (exact or (coverage >= 0.92 and (doi_present or metadata))):
        raise ValueError(f"Title not confirmed in document (token coverage={coverage})")
    evidence.update(
        {
            "identidade_confirmada": True,
            "titulo_completo_no_texto": exact,
            "cobertura_palavras_titulo": coverage,
            "doi_no_texto": doi_present,
            "excerto_inicio": text[:650],
            "estrutura_integral_validada": True,
        }
    )
    if metadata:
        evidence["metadados_pmc"] = {
            k: metadata.get(k)
            for k in (
                "pmcid",
                "doi",
                "title",
                "version",
                "is_manuscript",
                "is_retracted",
                "license_code",
            )
        }
    return fmt, evidence


def verify_record(record, doc):
    data = (ROOT / record["ficheiro"]).read_bytes()
    if sha256(data) != record["sha256"]:
        raise ValueError(f"{doc['id']}: SHA-256 mismatch")
    meta = None
    if source := record.get("metadados_fonte"):
        raw = (ROOT / source["ficheiro"]).read_bytes()
        if sha256(raw) != source["sha256"]:
            raise ValueError(f"{doc['id']}: source metadata SHA-256 mismatch")
        meta = json.loads(raw)
        if meta.get("pmcid") != doc["pmcid"] or compact(
            meta.get("doi") or ""
        ) != compact(doc["doi"]):
            raise ValueError(f"{doc['id']}: source metadata identity mismatch")
    fmt, _ = inspect_payload(data, doc, meta)
    if fmt != record["formato"] or len(data) != record["bytes"]:
        raise ValueError(f"{doc['id']}: format or size mismatch")


def write_reports(report):
    atomic_write(REPORT, json_bytes(report))
    rows = [
        "# Relatório de downloads do corpus",
        "",
        f"Execução terminada (UTC): {report['fim_utc']}",
        "",
        (
            f"Documentos: {report['resumo']['total']}; sucessos: {report['resumo']['sucessos']}; "
            f"falhas: {report['resumo']['falhas']}."
        ),
        "",
        (
            "O catálogo original foi preservado byte a byte. "
            f"SHA-256: `{report['catalogo']['sha256_antes']}`."
        ),
        "",
        (
            "Os nomes usam os IDs do catálogo. O formato é detetado pelo conteúdo. "
            "A identidade é confirmada no texto do documento; para PMC, também por "
            "DOI, PMCID e título nos metadados oficiais. PDFs: estrutura, número de "
            "páginas e texto das duas primeiras páginas; DOCX: integridade ZIP e "
            "texto integral; XML: identificadores JATS e corpo integral. "
            "Esta verificação confirma identidade e estrutura, sem auditar todas as "
            "tabelas, imagens ou resultados científicos."
        ),
        "",
        (
            "Os PDFs PMC provêm do serviço oficial NIH NLM NCBI PubMed Central "
            "Article Datasets, consultado nesta execução. "
            "[Documentação](https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/). "
            "Licenças específicas constam nos metadados de cada artigo. "
            "As páginas HTML de acesso não contam como documentos obtidos."
        ),
        "",
        (
            "O JSON associado contém URLs originais e finais, tentativas HTTP, "
            "data de aquisição, tamanho, SHA-256 e evidências de validação. "
            "Algumas URLs institucionais são temporárias; a página original "
            "permite localizar uma ligação atualizada."
        ),
        "",
        "Os ficheiros que já existiam antes da execução foram preservados.",
        "",
        "| ID | Estado | Formato | Ficheiro / motivo | SHA-256 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in report["documentos"]:
        detail = r.get("ficheiro") or r.get("erro", "")
        rows.append(
            f"| {r['id']} | {r['estado']} | {r.get('formato') or '-'} | "
            f"{detail.replace('|', '/')} | {r.get('sha256') or '-'} |"
        )
    rows.extend(
        [
            "",
            "## Incidentes e versões",
            "",
            (
                "As falhas abaixo pertencem a tentativas rejeitadas; o estado final de cada "
                "entrada consta na tabela anterior. Os detalhes e URLs de todas as tentativas "
                "estão preservados no relatório JSON."
            ),
            "",
        ]
    )
    for record in report["documentos"]:
        rejected = [
            a for a in record["tentativas"] if a.get("erro") or a.get("erro_validacao")
        ]
        if rejected:
            reasons = sorted({a.get("erro_validacao") or a["erro"] for a in rejected})
            rows.append(f"- {record['id']}: " + "; ".join(reasons))
        if record.get("formato") == "docx":
            rows.append(
                f"- {record['id']}: manuscrito aceite obtido no STORRE. "
                "O título do manuscrito tem pequenas diferenças em relação ao título "
                "publicado do catálogo; o DOI foi confirmado no próprio documento."
            )
    rows.extend(
        [
            "",
            "## Repetir ou verificar",
            "",
            (
                "Usar o ambiente Python do projeto (dependências já instaladas). "
                "O comando de descarga reutiliza os sucessos após os verificar e volta a "
                "tentar entradas em falta. A verificação offline não acede à rede."
            ),
            "",
            "```powershell",
            r".\.venv\Scripts\python.exe scripts/download_corpus.py",
            r".\.venv\Scripts\python.exe scripts/download_corpus.py --verify-only",
            "```",
        ]
    )
    atomic_write(
        OUTPUT / "relatorio_downloads.md", ("\n".join(rows) + "\n").encode("utf-8")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    original = CATALOGUE.read_bytes()
    catalogue = json.loads(original)
    docs = catalogue["documentos"]
    ids = [d["id"] for d in docs]
    if len(ids) != len(set(ids)) or any(
        not re.fullmatch(r"SPORT-\d{3}", i) for i in ids
    ):
        raise ValueError("Invalid or duplicate catalogue IDs")
    previous = json.loads(REPORT.read_bytes()) if REPORT.exists() else None
    if previous and previous["catalogo"]["sha256_antes"] != sha256(original):
        raise ValueError("Catalogue changed since the previous acquisition")
    old_records = {r["id"]: r for r in previous["documentos"]} if previous else {}
    records = {}
    pending = []
    for doc in docs:
        old = old_records.get(doc["id"])
        if old and old["estado"] == "sucesso":
            verify_record(old, doc)
            records[doc["id"]] = old
        else:
            pending.append(doc)
    if args.verify_only:
        if not previous or set(ids) != set(old_records):
            raise ValueError("Report does not cover all catalogue IDs")
        print(
            f"Verificados offline: {len(records)} documentos; {len(pending)} falhas registadas."
        )
        return 0
    report = {
        "schema_versao": 1,
        "inicio_utc": previous["inicio_utc"] if previous else now(),
        "fim_utc": None,
        "catalogo": {
            "ficheiro": CATALOGUE.relative_to(ROOT).as_posix(),
            "sha256_antes": sha256(original),
            "sha256_depois": None,
            "inalterado": None,
        },
        "ficheiros_preexistentes_preservados": (
            previous["ficheiros_preexistentes_preservados"]
            if previous
            else [
                {
                    "ficheiro": p.relative_to(ROOT).as_posix(),
                    "sha256": sha256(p.read_bytes()),
                }
                for p in OUTPUT.rglob("*")
                if p.is_file()
            ]
        ),
    }
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 8))) as pool:
        jobs = {pool.submit(Download(doc).run): doc for doc in pending}
        for job in as_completed(jobs):
            result = job.result()
            if old := old_records.get(result["id"]):
                result["tentativas"] = old.get("tentativas", []) + result["tentativas"]
            records[result["id"]] = result
            print(
                f"{result['id']}: {result['estado']} {result.get('formato') or result.get('erro')}",
                flush=True,
            )
    after = CATALOGUE.read_bytes()
    report["catalogo"].update(
        {"sha256_depois": sha256(after), "inalterado": after == original}
    )
    if after != original:
        raise ValueError("Catalogue was modified during acquisition")
    for saved in report["ficheiros_preexistentes_preservados"]:
        if sha256((ROOT / saved["ficheiro"]).read_bytes()) != saved["sha256"]:
            raise ValueError("A pre-existing file changed during acquisition")
    report["documentos"] = [records[i] for i in ids]
    counts = Counter(r["estado"] for r in records.values())
    report["resumo"] = {
        "total": len(docs),
        "sucessos": counts["sucesso"],
        "falhas": counts["falha"],
        "formatos": dict(
            Counter(r["formato"] for r in records.values() if r["estado"] == "sucesso")
        ),
    }
    report["fim_utc"] = now()
    write_reports(report)
    print(json.dumps(report["resumo"], ensure_ascii=True))
    return 1 if counts["falha"] else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
