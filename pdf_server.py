"""Serve the app's legal PDFs locally so source links can open in a new tab."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import quote, unquote, urlsplit

from config import DATA_PATH


_server = None


def _start_server():
    global _server
    if _server is not None:
        return _server

    pdf_dir = (DATA_PATH / "laws").resolve()

    class PdfHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            filename = unquote(urlsplit(self.path).path).lstrip("/")
            if not filename or Path(filename).name != filename:
                self.send_error(404)
                return

            pdf_path = (pdf_dir / filename).resolve()
            if pdf_path.parent != pdf_dir or pdf_path.suffix.lower() != ".pdf" or not pdf_path.is_file():
                self.send_error(404)
                return

            content = pdf_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Disposition", f'inline; filename="{pdf_path.name}"')
            self.send_header("Content-Length", str(len(content)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, _format, *_args):
            pass

    _server = ThreadingHTTPServer(("127.0.0.1", 0), PdfHandler)
    Thread(target=_server.serve_forever, daemon=True).start()
    return _server


def pdf_page_url(filename: str, page: int) -> str:
    """Return a local URL that opens a PDF at its 1-based page number."""
    server = _start_server()
    return f"http://127.0.0.1:{server.server_port}/{quote(filename)}#page={page}"
