"""Serve the app's legal PDFs locally so source links can open in a new tab."""

import atexit
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections import OrderedDict
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Thread
from urllib.parse import quote, unquote, urlsplit

from config import DATA_PATH


_server = None
_case_files = OrderedDict()


def _remove_case_files():
    for path, _filename in _case_files.values():
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


atexit.register(_remove_case_files)


def register_uploaded_pdf(pdf_bytes: bytes, filename: str) -> str:
    """Make an uploaded PDF available to local page-citation links."""
    token = hashlib.sha256(pdf_bytes).hexdigest()
    if token in _case_files:
        _case_files.move_to_end(token)
        return token

    safe_filename = Path(filename).name or "uploaded-case.pdf"
    with NamedTemporaryFile(prefix="legal-case-", suffix=".pdf", delete=False) as temp_file:
        temp_file.write(pdf_bytes)
        temp_path = Path(temp_file.name)
    _case_files[token] = (temp_path, safe_filename)

    while len(_case_files) > 8:
        _old_token, (old_path, _old_name) = _case_files.popitem(last=False)
        try:
            old_path.unlink(missing_ok=True)
        except OSError:
            pass
    return token


def uploaded_pdf_page_url(token: str, page: int) -> str:
    """Return a local URL that opens a registered uploaded PDF at a page."""
    server = _start_server()
    return f"http://127.0.0.1:{server.server_port}/case/{quote(token)}#page={page}"


def _start_server():
    global _server
    if _server is not None:
        return _server

    pdf_dir = (DATA_PATH / "laws").resolve()

    class PdfHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            parts = [unquote(part) for part in urlsplit(self.path).path.split("/") if part]
            display_filename = ""
            if len(parts) == 2 and parts[0] == "case":
                case_file = _case_files.get(parts[1])
                if case_file is None:
                    self.send_error(404)
                    return
                pdf_path, display_filename = case_file
            elif len(parts) == 1 and Path(parts[0]).name == parts[0]:
                display_filename = parts[0]
                pdf_path = (pdf_dir / display_filename).resolve()
                if pdf_path.parent != pdf_dir or pdf_path.suffix.lower() != ".pdf":
                    self.send_error(404)
                    return
            else:
                self.send_error(404)
                return

            if not pdf_path.is_file():
                self.send_error(404)
                return

            content = pdf_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header(
                "Content-Disposition",
                f"inline; filename*=UTF-8''{quote(display_filename or pdf_path.name)}",
            )
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
