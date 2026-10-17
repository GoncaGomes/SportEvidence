"""Offline regression checks for rejecting the wrong or incomplete payload."""

import importlib.util
import json
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pytest

SPEC = importlib.util.spec_from_file_location(
    "download_corpus",
    Path(__file__).resolve().parents[1] / "scripts/download_corpus.py",
)
acquisition = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acquisition)

DOC = {
    "id": "SPORT-001",
    "titulo": "Strength training and running economy",
    "doi": "10.1000/test.123",
    "pmcid": "PMC123",
}


def article_xml(*, doi=DOC["doi"], title=DOC["titulo"], body=True):
    content = f"<body><p>{'Full article text. ' * 400}</p></body>" if body else ""
    return (
        '<article article-type="review-article"><front><article-meta>'
        f'<article-id pub-id-type="doi">{doi}</article-id>'
        '<article-id pub-id-type="pmcid">PMC123</article-id>'
        f"<title-group><article-title>{title}</article-title></title-group>"
        f"</article-meta></front>{content}</article>"
    ).encode()


def test_full_article_identifiers_and_body_are_verified():
    fmt, evidence = acquisition.inspect_payload(article_xml(), DOC)
    assert fmt == "xml"
    assert evidence["identidade_confirmada"]


@pytest.mark.parametrize(
    "payload",
    [
        b"<html><title>403 Forbidden</title></html>",
        article_xml(doi="10.1000/wrong-document"),
        article_xml(title="Unrelated study"),
        article_xml(body=False),
    ],
)
def test_block_pages_wrong_documents_and_abstracts_are_rejected(payload):
    with pytest.raises(ValueError):
        acquisition.inspect_payload(payload, DOC)


def test_title_only_in_references_does_not_identify_xml_article():
    payload = article_xml(title="Unrelated study").replace(
        b"</article>", f"<back><ref>{DOC['titulo']}</ref></back></article>".encode()
    )
    with pytest.raises(ValueError, match="Title not confirmed"):
        acquisition.inspect_payload(payload, DOC)


def test_truncated_pdf_is_rejected():
    with pytest.raises(ValueError, match="end-of-file"):
        acquisition.inspect_payload(b"%PDF-1.7\npartial download", DOC)


def test_truncated_docx_is_rejected():
    stream = BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr("word/document.xml", "<document/>")
    with pytest.raises(BadZipFile):
        acquisition.inspect_payload(stream.getvalue()[:-3], DOC)


def test_local_checksum_detects_changed_download(tmp_path, monkeypatch):
    monkeypatch.setattr(acquisition, "ROOT", tmp_path)
    path = tmp_path / "SPORT-001.xml"
    original = article_xml()
    path.write_bytes(original + b" ")
    record = {"ficheiro": path.name, "sha256": acquisition.sha256(original)}
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        acquisition.verify_record(record, DOC)


def test_source_metadata_must_match_catalogue(tmp_path, monkeypatch):
    monkeypatch.setattr(acquisition, "ROOT", tmp_path)
    payload = article_xml()
    path = tmp_path / "SPORT-001.xml"
    path.write_bytes(payload)
    meta = json.dumps({"pmcid": "PMC999", "doi": DOC["doi"]}).encode()
    (tmp_path / "source.json").write_bytes(meta)
    record = {
        "ficheiro": path.name,
        "sha256": acquisition.sha256(payload),
        "metadados_fonte": {
            "ficheiro": "source.json",
            "sha256": acquisition.sha256(meta),
        },
    }
    with pytest.raises(ValueError, match="identity mismatch"):
        acquisition.verify_record(record, DOC)
