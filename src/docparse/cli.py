"""Command line.

  python -m docparse invoice.pdf receipt.jpg --out results/
  python -m docparse scans/ --out results/ --markdown
  python -m docparse --download-models      # pre-fetch models (Docker build / offline)
"""
import argparse
import json
import sys
from pathlib import Path

SUPPORTED = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}


def collect(inputs):
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            yield from sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED)
        elif path.exists():
            yield path
        else:
            print(f"skip: {path} not found", file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="docparse", description="Parse PDFs and images to JSON.")
    ap.add_argument("inputs", nargs="*", help="files or folders")
    ap.add_argument("--out", default="output", help="output folder (default: output)")
    ap.add_argument("--pages", help="pages to read, e.g. 1-3,5")
    ap.add_argument("--markdown", action="store_true", help="also write a .md file")
    ap.add_argument("--download-models", action="store_true", help="load all models and exit")
    args = ap.parse_args(argv)

    from .parser import DocumentParser
    from .utils import to_markdown

    parser = DocumentParser()
    if args.download_models:
        parser.models.warmup()
        print("models ready")
        return 0
    if not args.inputs:
        ap.error("give at least one file or folder")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    failed = 0
    for path in collect(args.inputs):
        try:
            result = parser.parse(path.read_bytes(), path.name, args.pages)
        except Exception as e:  # keep going with the other files
            failed += 1
            print(f"FAIL {path}: {e}", file=sys.stderr)
            continue
        (out / f"{path.stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
        if args.markdown:
            (out / f"{path.stem}.md").write_text(to_markdown(result), "utf-8")
        routes = sorted({r for p in result["pages"] for r in p["routes"]})
        conf = result["confidence"]
        print(f"ok   {path.name}: {result['page_count']} page(s), "
              f"{result['elapsed_ms'] / 1000:.1f}s, confidence={conf['overall']} (ocr={conf['ocr']}), "
              f"low_lines={conf['low_confidence_lines']}, routes={','.join(routes)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
