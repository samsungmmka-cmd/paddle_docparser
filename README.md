# docparse: PaddleOCR document parser

Turns PDFs and images into structured JSON (or Markdown): text, headings,
tables, formulas, handwriting. Every page is routed automatically to the
cheapest pipeline that can read it, so simple pages stay fast and only the
hard parts pay for the heavy models.

## How routing works

```
file ──► PDF? ──yes──► page has a real text layer? ──yes──► read text + tables from PDF (no OCR, ~0.1 s)
  │                         │                                 ├─ big embedded image  → OCR that region only
  │                         no (scanned)                      └─ borderless table → structure model on that region only
  │                         ▼
  └──── image ─────► fix orientation ─► layout check (small model)
                                          ├─ table / formula / seal found → PP-StructureV3, only those modules on
                                          └─ plain text → fast OCR → re-read only low-confidence lines with the
                                                                      strong model (handwriting, faded print)
```

Nothing is run "just in case": handwriting re-reads only the weak lines, not
the whole page.

**Charts:** chart *data* extraction (PP-Chart2Table) has been removed. It was very
slow and could hang the whole service. A chart now comes back as an `image` block
(`label: "chart"`) and any text printed on it (titles, axis labels) as normal text.

## Quick start (local, CPU)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python -m docparse invoice.pdf receipt.jpg --out output/ --markdown

export DOCPARSE_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
uvicorn docparse.api:app --port 8080
```

The first run downloads the models (a few hundred MB) into `~/.paddlex`.
The CLI needs no API key; the HTTP API refuses to start without one.

## Docker

```bash
cp .env.example .env      # then put a key in DOCPARSE_API_KEY (letters/digits only, no $)
docker compose up --build -d
docker compose logs -f ocr   # wait for "Uvicorn running on http://0.0.0.0:8080"
```

The service is **private**: its port is published on `127.0.0.1` only, so it can't be
reached from other machines. Expose it through nginx on the same server: paste the
`location` blocks from `deploy/nginx/docparse.conf` into your `server { }` block, run
`nginx -t`, then reload nginx. Public URLs become `https://<gateway>/ocr/v1/parse` and
`https://<gateway>/ocr/health`.

Compose reads `.env`; `.env.example` is only a template. After changing `.env`, run
`docker compose up -d` again so the container is recreated with the new values.
Don't use `$` in `.env` values: Compose treats it as a variable and drops part of the value.

`Dockerfile.gpu` is still included if you move to a GPU server later.

## API

`POST /v1/parse` with a multipart `file` field and the header `X-API-Key: <your key>`.
Requests without the right key get `401`.

| Query param | Default | Meaning |
|---|---|---|
| `pages` | all | e.g. `1-3,5`. At most 100 pages per request (`DOCPARSE_MAX_PAGES`) |
| `output` | `json` | `json` or `markdown` |
| `include_lines` | `false` | include each text line with its own box and score |

```bash
curl -H "X-API-Key: $DOCPARSE_API_KEY" -F "file=@invoice.pdf" \
     "http://localhost:8080/v1/parse?pages=1-2"
```

| Status | When |
|---|---|
| `401` | missing or wrong API key |
| `400` | empty file or invalid `pages` value |
| `413` | file over `max_upload_mb`, or more than 100 pages selected (pick a range with `pages`) |
| `415` | not a PDF or supported image |

`GET /health` (no key needed) returns `{"status": "ok", "ready": true}` once models are loaded.

### Output

```json
{
  "filename": "invoice.pdf", "file_type": "pdf", "page_count": 2,
  "confidence": {"overall": 0.947, "ocr": 0.904, "low_confidence_lines": 1, "handwriting_pages": [2]},
  "text": "=== Page 1 ===\nINVOICE 2026-0042\n...\n\n| Item | Qty | Price |\n| Paper | 2 | 10.00 |\n\n=== Page 2 ===\n...",
  "pages": [{
    "page": 1, "width": 1654, "height": 2339,
    "source": "text_layer", "routes": ["text_layer", "pdf_tables"],
    "orientation": 0, "handwriting_likely": false,
    "confidence": 1.0,
    "blocks": [
      {"type": "text",  "bbox": [200, 90, 1004, 212], "text": "INVOICE 2026-0042\n...", "confidence": 1.0},
      {"type": "table", "bbox": [200, 522, 487, 672], "rows": [["Item","Qty","Price"], ["Paper","2","10.00"]],
       "cell_confidence": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], "confidence": 1.0}
    ],
    "elapsed_ms": 80.1
  }],
  "elapsed_ms": 2140.5
}
```

### Confidence

| Field | Meaning |
|---|---|
| `confidence.overall` | score for all text in the document (0–1) |
| `confidence.ocr` | score for OCR'd text only; `null` when nothing needed OCR. Use this one to judge OCR quality: `overall` looks high on mostly-digital PDFs |
| `confidence.low_confidence_lines` | OCR lines and table cells scoring below `DOCPARSE_HANDWRITING_THRESHOLD` (0.80) after the re-read |
| `confidence.handwriting_pages` | page numbers that look handwritten |
| `pages[].confidence` | score for that page |
| `blocks[].confidence` | score for every block type: text, heading, table, image, formula… |
| `blocks[].cell_confidence` | tables only: one score per cell, same shape as `rows` |

- Scores are averages **weighted by character count**, so a long line counts more than a single character.
- Text read from a PDF's text layer is exact and scores `1.0`.
- `null` means there is no text to score (an image with no text in it, an empty cell, a blank page),
  or a table cell whose text could not be matched to an OCR line.
- Text printed on an image or chart becomes that image block's `text`, and its score is the image's `confidence`.

`text` holds the whole document as one string in reading order: pages start with
`=== Page N ===`, tables are written as `| a | b |` rows and images as `[image]`.

- **Block types:** `title`, `heading`, `text`, `table` (`rows`, plus `html` from OCR),
  `formula` (`latex`), `image` (photos, figures and charts), `caption`, `header`, `footer`, `seal`.
- **Coordinates** are pixels: of the image as uploaded (images), or of the page rendered
  at `render_dpi` (PDFs). Very large PDF pages are rendered smaller so one page never
  exceeds `max_render_pixels` (25 MP); the page's `width`/`height` always match the boxes.
  If a page was rotated upright, boxes are in the upright frame.
- **`source`** says where text came from: `text_layer`, `ocr`, `structure`, `embedded_image_ocr`.

### Calling from .NET

```csharp
using var form = new MultipartFormDataContent();
form.Add(new StreamContent(File.OpenRead("invoice.pdf")), "file", "invoice.pdf");
http.DefaultRequestHeaders.Add("X-API-Key", apiKey);   // from your app's secret store
var res = await http.PostAsync("http://ocr:8080/v1/parse", form);
var json = await res.Content.ReadAsStringAsync();
```

## Configuration

All settings are environment variables prefixed with `DOCPARSE_` (see
`src/docparse/config.py` and `.env.example`). The ones you are most likely to change:

| Variable | Default | Notes |
|---|---|---|
| `DOCPARSE_API_KEY` | none (required) | at least 16 characters; sent by clients as `X-API-Key` |
| `DOCPARSE_DEVICE` | `cpu` | `gpu:0` for GPU |
| `DOCPARSE_CPU_THREADS` | cores, max 8 | match physical cores |
| `DOCPARSE_MAX_PAGES` | `100` | pages processed per request |
| `DOCPARSE_MAX_RENDER_PIXELS` | `25000000` | size cap for one rendered PDF page |
| `DOCPARSE_ENABLE_FORMULAS` / `_SEALS` | on / off | turn off what you never need |
| `DOCPARSE_HANDWRITING_THRESHOLD` | `0.80` | lines scoring below this are re-read |
| `DOCPARSE_HANDWRITING_PAGE_RATIO` | `0.30` | share of weak lines that marks a page handwritten |
| `DOCPARSE_MIN_TEXT_LAYER_CHARS` | `50` | fewer → treat PDF page as scanned |
| `DOCPARSE_RENDER_DPI` | `200` | 200–300 for small print |

## Performance (rough, per page)

| Page | CPU (8 cores) | GPU |
|---|---|---|
| Digital PDF page | ~0.1 s | ~0.1 s |
| Scanned printed text | 2–4 s | < 1 s |
| Scanned with tables | 4–8 s | ~1 s |
| Handwritten | 3–6 s | ~1 s |

Tips: use the GPU image for volume; run one worker per container and scale by adding containers (Paddle is not thread-safe,
so requests inside one process are handled one at a time).

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Tests use a fake model, so they check the routing and output format in about a second
without downloading anything. Before going live, run a sample of your real documents
through the CLI and look at the `routes` and `confidence` values.

## Known limits

- **Check rotation once:** if sideways scans come out with garbage text, the rotation
  direction for your model version is reversed; flip `ROTATE_TURNS` in `parser.py`
  (90 ↔ 270).
- Table cells that span several rows (`rowspan`) are not expanded in `rows`; the `html`
  field keeps the full structure.
- Very messy handwriting is better served by PaddleOCR-VL on a GPU.
