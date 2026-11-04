import subprocess
import sys


def test_utf8_stdio_lets_redirected_output_print_box_chars(tmp_path):
    # Windows: output dialihkan -> cp1252; tabel rich rfdetr ('│') dulu membuat training crash
    code = ("from cascade_sam3.cli import utf8_stdio; utf8_stdio(); "
            "from rich.console import Console; Console().print('Val \u2502 Overall'); print('OK')")
    out = tmp_path / "o.txt"
    with open(out, "wb") as f:
        r = subprocess.run([sys.executable, "-c", code], stdout=f, stderr=subprocess.STDOUT)
    assert r.returncode == 0, out.read_bytes()
    assert b"OK" in out.read_bytes()
