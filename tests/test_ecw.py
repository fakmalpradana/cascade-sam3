import subprocess

import pyproj

from cascade_sam3.data import ecw, inspect_data


def test_inspect_ecw_fallback_crs_is_epsg(tmp_path, monkeypatch):
    """Raster yang hanya terbaca lewat gdalinfo QGIS harus memberi CRS 'EPSG:n' yang bisa dipakai label."""
    f = tmp_path / "a.ecw"
    f.write_bytes(b"x")
    info = {"driverShortName": "ECW", "size": [10, 10], "geoTransform": [0, 0.05, 0, 0, 0, -0.05],
            "coordinateSystem": {"wkt": pyproj.CRS("EPSG:32650").to_wkt()}, "bands": [{}, {}, {}, {}],
            "cornerCoordinates": {"upperLeft": [0, 1], "lowerRight": [1, 0], "upperRight": [1, 1],
                                  "lowerLeft": [0, 0]}}
    monkeypatch.setattr(ecw, "gdalinfo_json", lambda cfg, p: info)
    d = inspect_data._raster_info({}, f)
    assert d["readable"] is False and d["via"] == "gdalinfo QGIS"
    assert d["crs"] == "EPSG:32650"
    assert inspect_data._work_crs([d]) == "EPSG:32650"


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
