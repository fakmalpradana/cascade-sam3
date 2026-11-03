# HANDOFF — lanjut di PC Windows

Kode PLAN.md fase 0–7 ditulis di Mac (commit awal `00bf1b8`). Di Mac: 28 tes lulus, `ruff` bersih,
CI hijau (ubuntu + windows, CPU, tanpa extra `train`). **Belum pernah jalan di PC ini**: belum ada
`.venv`, GPU, ECW, data `Y:`, bobot SAM 3 asli, maupun `rfdetr`. Status user: baru `git clone`.

Baca dulu: `PLAN.md` (rencana + fakta teknis §2), `CLAUDE.md` (aturan repo), `README.md` (urutan skrip).

## Mesin

- Windows, RTX 3060 12 GB, RAM 64 GB, user `GeoAI_INT_4`, repo di `C:\Users\GeoAI_INT_4\cascade-sam3`.
- Data di `Y:` (ECW + SHP/GDB), work_dir default `D:/cascade-sam3-work` (cek ruang > 150 GB).
- Akun bukan admin dianggap default; minta user bila perlu elevasi.

## Status per bagian

| Bagian | Status | Yang harus dicek pertama di sini |
|---|---|---|
| `setup_env.ps1` | belum pernah jalan | `py -3.12` ada? Execution policy? torch cu128 terpasang dan `cuda.is_available()` True? `pip install -e .[train,dev]` resolve? |
| `check-env --smoke` | belum | login HF (`.venv\Scripts\hf.exe auth login`), akses gated `facebook/sam3`, GDAL QGIS + driver ECW |
| `inspect` / `convert-ecw` | belum | path `Y:` di `configs/local.yaml`, auto-deteksi `gdal_translate` QGIS (`data.gdal_bin`), ruang disk |
| `prepare` | teruji sintetis saja | label_filter & CRS nyata — tunjukkan ke user sebelum lanjut |
| `train-detector` (RF-DETR) | **API belum pernah dieksekusi** | `rfdetr==1.11.2`: `RFDETRSegLarge`, `train(**kw)`, `from_checkpoint`, `predict` di `models/detector.py` & `train/train_detector.py`; resolusi kelipatan 24 |
| `train-sam3` | teruji model mini CPU | VRAM 12 GB dengan bobot asli, bf16 encoder + fp32 decoder |
| `ConceptSegmenter` (prompt teks) | API dicocokkan dengan kode transformers 5.22.1, belum dengan bobot | `get_text_features` → `text_embeds=`; tokenizer perlu unduhan |
| skrip `.ps1` lain | belum pernah jalan | `_common.ps1` (`Invoke-Cli`, `Start-Log`), transcript ke `work_dir/logs/` |

## Cara kerja

1. Satu masalah per langkah: jalankan, baca error, perbaiki minimal, ulang. Jangan loncat fase.
2. Setelah perbaikan kode: `.venv\Scripts\python.exe -m pytest` dan `.venv\Scripts\ruff.exe check .`,
   lalu commit + push ke `main` (pesan jelas). Mac akan `git pull`.
3. Ikuti "Lulus jika" tiap fase di PLAN.md §7.
4. Perubahan `configs/local.yaml` (label_filter, coverage_mode, target_gsd_m) tunjukkan ke user dulu.
5. Jangan commit data, bobot, raster, `configs/local.yaml`. Jangan setel ambang pakai area uji.
6. Jangan ketik password/token ke mana pun; user sendiri yang menjalankan `hf auth login`.
7. Jawab dalam Bahasa Indonesia, ringkas dan teknis.

## Checklist user (PLAN.md §13)

- [ ] Akses Hugging Face `facebook/sam3` disetujui
- [ ] QGIS LTR terpasang
- [ ] Drive kerja > 150 GB kosong
- [ ] Bobot/skrip SAM-YOLO lama untuk baseline Fase 3
- [ ] Format output & CRS target
