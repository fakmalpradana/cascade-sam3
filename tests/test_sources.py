from cascade_sam3.data import sources

CFG = {"data": {"vector_exts": [".gpkg", ".shp"]}}


def test_label_name_regex_filters_by_filename(tmp_path):
    names = ["AF-08-C_BO.gpkg", "AF-08-C_RO.gpkg", "AF-08-C_KANOPI.gpkg", "AL-30-A-BO.gpkg",
             "BA-17-B_BO_TAMBAL.gpkg", "BE-23-B_BO_1.gpkg", "bb-14-a_bo.GPKG", "BF-17-D_ RO.gpkg"]
    for n in names:
        d = tmp_path / n.split("_")[0].split(".")[0]
        d.mkdir(exist_ok=True)
        (d / n).write_bytes(b"")

    allf = sources.vector_files({"labels": str(tmp_path)}, CFG)
    assert len(allf) == len(names)

    city = {"labels": str(tmp_path), "label_name_regex": r"[-_]BO(_TAMBAL|_\d+)?\.gpkg$"}
    got = sorted(p.name for p in sources.vector_files(city, CFG))
    assert got == sorted(["AF-08-C_BO.gpkg", "AL-30-A-BO.gpkg", "BA-17-B_BO_TAMBAL.gpkg",
                          "BE-23-B_BO_1.gpkg", "bb-14-a_bo.GPKG"])
