import subprocess

from cascade_sam3.data import ecw


def test_gdal_run_windows_cmd_quoting(tmp_path, monkeypatch):
    """Path QGIS berspasi harus sampai ke cmd.exe tanpa escape \\" (list2cmdline)."""
    bindir = tmp_path / "QGIS 3.40" / "bin"
    bindir.mkdir(parents=True)
    (bindir / "o4w_env.bat").write_text("")
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(ecw, "IS_WIN", True)
    monkeypatch.setattr(ecw.subprocess, "run", fake_run)
    cfg = {"data": {"gdal_bin": str(bindir)}}
    ecw.gdal_run(cfg, "gdalinfo", ["-json", "C:/data dir/a.ecw"])

    cmd = seen["cmd"]
    assert isinstance(cmd, str), "harus string mentah agar Python tidak menambah escape"
    assert '\\"' not in cmd
    # /q: echo off, agar baris perintah o4w_env.bat tidak mencemari stdout (gdalinfo -json)
    assert cmd.startswith('cmd /s /q /c "call "')
    assert f'"{bindir / "o4w_env.bat"}"' in cmd
    assert '"C:/data dir/a.ecw"' in cmd
    assert cmd.endswith('"')
