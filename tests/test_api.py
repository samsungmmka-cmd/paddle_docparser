import pytest
from fastapi.testclient import TestClient

from conftest import API_KEY
from docparse import DocumentParser, Settings
from docparse.api import create_app

AUTH = {"X-API-Key": API_KEY}


@pytest.fixture
def client(parser):
    with TestClient(create_app(parser, warmup=False)) as c:
        yield c


def test_parse_endpoint(client, native_pdf):
    r = client.post("/v1/parse", headers=AUTH, files={"file": ("invoice.pdf", native_pdf, "application/pdf")})
    assert r.status_code == 200
    body = r.json()
    assert body["file_type"] == "pdf"
    assert all("lines" not in b for b in body["pages"][0]["blocks"])  # trimmed by default

    md = client.post("/v1/parse?output=markdown", headers=AUTH, files={"file": ("invoice.pdf", native_pdf)})
    assert "| Item | Qty | Price |" in md.text


def test_rejects_unknown_files(client):
    r = client.post("/v1/parse", headers=AUTH, files={"file": ("notes.docx", b"PK\x03\x04xxxx")})
    assert r.status_code == 415


# ---------- H3: API key ----------

@pytest.mark.parametrize("headers", [{}, {"X-API-Key": ""}, {"X-API-Key": "wrong-key-0123456789"},
                                     {"X-API-Key": API_KEY + "x"}, {"Authorization": f"Bearer {API_KEY}"}])
def test_requests_without_the_right_key_are_rejected(client, native_pdf, fake, headers):
    r = client.post("/v1/parse", headers=headers, files={"file": ("invoice.pdf", native_pdf)})
    assert r.status_code == 401
    assert fake.calls == []  # rejected before any parsing work


def test_health_needs_no_key(client):
    assert client.get("/health").json()["status"] == "ok"


@pytest.mark.parametrize("key", ["", "short-key"])
def test_service_refuses_to_start_without_a_strong_key(fake, key):
    app = create_app(DocumentParser(Settings(api_key=key), models=fake), warmup=False)
    with pytest.raises(RuntimeError, match="DOCPARSE_API_KEY"):
        with TestClient(app):
            pass


def test_api_key_is_not_shown_in_settings_repr():
    assert API_KEY not in repr(Settings(api_key=API_KEY))


# ---------- H1: page limits over HTTP ----------

def test_page_errors_are_client_errors(client, native_pdf):
    bad = client.post("/v1/parse?pages=abc", headers=AUTH, files={"file": ("a.pdf", native_pdf)})
    assert bad.status_code == 400
    huge = client.post("/v1/parse?pages=1-10000000000", headers=AUTH, files={"file": ("a.pdf", native_pdf)})
    assert huge.status_code == 200 and huge.json()["page_count"] == 1


def test_more_than_100_pages_is_rejected(client):
    import io
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for _ in range(101):
        c.drawString(72, 800, "page")
        c.showPage()
    c.save()
    r = client.post("/v1/parse", headers=AUTH, files={"file": ("big.pdf", buf.getvalue())})
    assert r.status_code == 413 and "limit is 100" in r.json()["detail"]
    ok = client.post("/v1/parse?pages=1-100", headers=AUTH, files={"file": ("big.pdf", buf.getvalue())})
    assert ok.status_code == 200 and len(ok.json()["pages"]) == 100
