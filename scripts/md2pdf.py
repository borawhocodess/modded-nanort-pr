"""Markdown (com diagramas mermaid) -> PDF.

Mesma rota do PDF original de docs/: HTML renderizado no Chrome headless. pandoc
converte o corpo; os blocos ```mermaid saem antes (pandoc os viraria <pre><code>,
que o mermaid.js ignora) e voltam como <pre class="mermaid">. O mermaid.min.js é
embutido inline para o Chrome renderizar sem depender de rede na hora do print.

    pixi run python scripts/md2pdf.py docs/arquivo.md
    pixi run python scripts/md2pdf.py docs/arquivo.md docs/saida.pdf

Requer: pandoc, Google Chrome. O mermaid.min.js é baixado uma vez e cacheado em
~/.cache/nanort/.
"""

import html
import os
import re
import ssl
import subprocess
import sys
from pathlib import Path
from urllib import request

import certifi

MERMAID_VERSION = "11.16.1"
MERMAID_URL = (
    f"https://cdn.jsdelivr.net/npm/mermaid@{MERMAID_VERSION}/dist/mermaid.min.js"
)
CHROME = Path(
    os.environ.get(
        "NANORT_CHROME",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    )
)


def mermaid_js() -> str:
    """mermaid.min.js, baixado uma vez e cacheado. O bundle termina em
    globalThis["mermaid"] = ..., então funciona como <script> clássico."""
    cache = Path.home() / ".cache" / "nanort" / f"mermaid-{MERMAID_VERSION}.min.js"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"[md2pdf] baixando {MERMAID_URL}")
        tls_context = ssl.create_default_context(cafile=certifi.where())
        with request.urlopen(MERMAID_URL, timeout=60, context=tls_context) as response:
            cache.write_bytes(response.read())
    return cache.read_text()


if len(sys.argv) not in {2, 3}:
    raise SystemExit(f"usage: {Path(sys.argv[0]).name} INPUT.md [OUTPUT.pdf]")

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) == 3 else SRC.with_suffix(".pdf")
if not SRC.is_file():
    raise FileNotFoundError(f"Markdown source not found: {SRC}")
if not CHROME.is_file():
    raise FileNotFoundError(
        f"Chrome executable not found: {CHROME}. Set NANORT_CHROME to its path."
    )
OUT.parent.mkdir(parents=True, exist_ok=True)

md = SRC.read_text(encoding="utf-8")

# 1. tira os blocos mermaid, deixando marcadores que sobrevivem ao pandoc
blocks = []


def _stash(m):
    blocks.append(m.group(1))
    return f"\n\nPLACEHOLDERMERMAID{len(blocks) - 1}ENDPLACEHOLDER\n\n"


md_nomermaid = re.sub(r"```mermaid\n(.*?)```", _stash, md, flags=re.DOTALL)
print(f"[md2pdf] {len(blocks)} diagramas mermaid extraídos")

# 2. pandoc -> fragmento HTML
body = subprocess.run(
    ["pandoc", "-f", "gfm", "-t", "html5", "--no-highlight"],
    input=md_nomermaid,
    capture_output=True,
    text=True,
    check=True,
).stdout

# 3. devolve os diagramas
for i, blk in enumerate(blocks):
    esc = blk.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for pat in (
        f"<p>PLACEHOLDERMERMAID{i}ENDPLACEHOLDER</p>",
        f"PLACEHOLDERMERMAID{i}ENDPLACEHOLDER",
    ):
        if pat in body:
            body = body.replace(pat, f'<pre class="mermaid">{esc}</pre>')
            break
    else:
        raise SystemExit(f"marcador {i} perdido pelo pandoc")

if "PLACEHOLDERMERMAID" in body:
    raise RuntimeError("marcador mermaid residual no HTML renderizado")

CSS = """
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
       font-size: 10.5pt; line-height: 1.55; color: #1a1a1a; margin: 0; }
h1 { font-size: 20pt; margin: 0 0 .6em; letter-spacing: -.01em; }
h2 { font-size: 14pt; margin: 1.7em 0 .5em; padding-bottom: .25em;
     border-bottom: 1px solid #d8d8d8; break-after: avoid; }
h1, h2, h3 { break-after: avoid-page; }
p, li { orphans: 3; widows: 3; }
code, pre { font-family: ui-monospace, "SF Mono", Menlo, monospace; }
code { font-size: 9pt; background: #f2f2f4; padding: .12em .3em; border-radius: 3px; }
pre { background: #f7f7f9; border: 1px solid #e3e3e7; border-radius: 5px;
      padding: .7em .9em; font-size: 8.3pt; line-height: 1.45;
      white-space: pre-wrap; word-break: break-word; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: inherit; }
table { border-collapse: collapse; width: 100%; margin: 1em 0;
        font-size: 9.2pt; break-inside: avoid; }
th, td { border: 1px solid #dcdce0; padding: .42em .6em; text-align: left;
         vertical-align: top; }
th { background: #f2f2f4; font-weight: 600; }
blockquote { margin: 1em 0; padding: .55em .9em; background: #f8f8fa;
             border-left: 3px solid #b8b8c0; color: #333; break-inside: avoid; }
blockquote p { margin: .25em 0; }
hr { border: 0; border-top: 1px solid #e3e3e7; margin: 1.8em 0; }
pre.mermaid { background: none; border: 0; padding: 0; text-align: center;
              break-inside: avoid; margin: 1.2em 0; }
pre.mermaid svg { max-width: 100%; height: auto; }
/* nao quebrar identificadores no meio ("RelationalTransformer.forw|ard") */
pre.mermaid .nodeLabel, pre.mermaid .edgeLabel, pre.mermaid foreignObject div {
  word-break: keep-all !important; overflow-wrap: normal !important; }
"""

html_lang = "pt-BR" if SRC.stem.endswith("_pt") else "en"
html = f"""<!doctype html>
<html lang="{html_lang}"><head><meta charset="utf-8">
<title>{html.escape(SRC.stem)}</title>
<style>{CSS}</style>
<script>{mermaid_js()}</script>
</head><body>
{body}
<script>
mermaid.initialize({{ startOnLoad: false, theme: "neutral", fontSize: 15,
                      flowchart: {{ useMaxWidth: true, htmlLabels: true,
                                    nodeSpacing: 22, rankSpacing: 30, wrappingWidth: 260,
                                    padding: 6 }} }});
mermaid.run().then(() => {{ document.title = "READY-" + document.title; }});
</script>
</body></html>"""

tmp_html = OUT.with_suffix(".build.html")
tmp_html.write_text(html, encoding="utf-8")
print(f"[md2pdf] HTML: {tmp_html} ({len(html)} bytes)")

try:
    subprocess.run(
        [
            str(CHROME),
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--virtual-time-budget=20000",
            "--run-all-compositor-stages-before-draw",
            f"--print-to-pdf={OUT.resolve()}",
            tmp_html.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
    )
finally:
    tmp_html.unlink(missing_ok=True)
print(f"[md2pdf] PDF: {OUT} ({OUT.stat().st_size} bytes)")
