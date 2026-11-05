# HANDOFF — pindah ke PC Windows ke-2 (4 Nov 2026, malam)

PC lama (`DESKTOP-HIO7EQ9`) dipakai bersama aplikasi desktop (±3,5 GB VRAM terpakai) → training lambat.
PC baru: spec sama (RTX 3060 12 GB, RAM 64 GB), **tanpa proses lain**, akses NAS sama.

Baca dulu: `PLAN.md` (rencana, "Lulus jika" §7), `CLAUDE.md` (aturan repo), `README.md`. Jawab dalam Bahasa
Indonesia, ringkas, gaya terse.

## Status fase

| Fase | Status | Bukti / catatan |
|---|---|---|
| 0 Lingkungan | **lulus** (PC lama) | check-env 8/8 OK, CI hijau. PC baru: perlu setup ulang (lihat bawah) |
| 1 Inspect + convert-ecw | **lulus** | GeoTIFF di `work_dir/ortho_converted/{dki,bontang}/` (JPEG, mask, overview 2–64) |
| 2 prepare | **lulus** | 6 area uji (3 kepadatan/kota), quicklook & overlay dicek, 0 tile train/valid beririsan area uji |
| 3 Baseline | **lulus** | `sam3_zeroshot`: F1 objek 0,740 · IoU piksel 0,823 · boundary F1 0,729 · AP50 0,663 (README) |
| 4 train-detector | **lulus** (PC baru, 5 Nov) | stop di epoch 2 (keputusan user): F1 objek 0,802 ≥ 0,740. Bobot: `runs/detector/checkpoint_best_ema_ep2_stop.pth` |
| 5 train-sam3 | **belum lulus** (keputusan user 5 Nov; target baru IoU > 0,90) | Tahap 1: detektor 12 epoch, decoder 6 epoch (mIoU val 0,885). Area uji, ambang bawaan: `detector` IoU 0,8847, boundary F1 0,8004; `cascade` IoU 0,8856, boundary F1 0,7927 (DKI 0,791 → 0,769). Ambang hasil tuning tidak membantu di area uji (README). Tahap 2 menunggu keputusan user |

Opsi baseline C (keputusan user): SAM-YOLO tidak ada → baseline = SAM 3 zero-shot.
Kriteria Fase 4: F1 objek detector ≥ 0,740 pada area uji. Fase 5: mIoU val naik dari zero-shot, dan
`cascade` > `detector` pada boundary F1 / IoU piksel.

## Data (semua di NAS, sudah jadi — JANGAN diulang)

- `work_dir` = `//server_12/Server_12/2026/25 RND/cascade-sam3-work` (W: = `\\server_12\Server_12`)
- Sumber: Y: = `\\Server_11\DATA_Server11` (ECW + label). Hanya perlu bila mengulang Fase 1–2.
- `catalog/`, `testsets/` (6 area), `dataset/coco/{train,valid,test}` (train 11.431, valid 2.311, test 437
  tile 1008 px @ 0,08 m), `dataset/coco_train6000/` (subset train 6.000, file_name relatif ke `coco/`),
  `predictions/sam3_zeroshot/`, `reports/validation_sam3_zeroshot.md`.
- Run detektor: `runs/detector/seg-large/`. Log konsol: `logs/train_detector_console.log`.

## Fase 4 — keadaan saat pindah

Setelan (configs/local.yaml): seg-large, resolusi 504, 6.000 tile train, batch 1 × grad_accum 16,
`checkpoint_interval: 1`, 60 epoch maks + early stopping (patience 12), VRAM dibatasi 70%.
Kecepatan PC lama: 1,25 it/s → train 80 mnt + validasi 18 mnt ≈ **1,7 jam/epoch**.

Run terakhir mulai ±20:35. Epoch 0 selesai ±22:15 → `checkpoint_0.ckpt` (state lengkap untuk resume)
**bila PC lama dihentikan setelah itu**. Cek isi `runs/detector/seg-large/` di PC baru.

**Masalah yang sudah diperbaiki (jangan diulang):**
1. VRAM penuh di Windows tidak OOM tapi meluber ke RAM (sysmem fallback) → macet. Fix: `max_vram_fraction`
   (`utils/gpu.py`, commit 16769ff).
2. OOM di loss mask: group DETR (13 grup) × tile padat (maks 168 bangunan) → 2,4 GB sekali alokasi. Fix:
   `train/rfdetr_patches.py` membungkus `_sample_target_masks_at_points` per potongan 256 (hasil identik,
   puncak 3,1 → 0,46 GB; commit 49fcb9c).
3. Crash akhir validasi: tabel `rich` ('│') ke stdout cp1252. Fix: CLI paksa UTF-8 (commit 9e736fa).
4. `--resume` rfdetr 1.11.2 pakai `last.ckpt` / `checkpoint_<n>.ckpt` (commit 0c20edc).
5. Log tidak menampilkan progress → `detector.extra: {progress_bar: tqdm}`.

## Langkah di PC baru

1. **Pastikan training di PC lama sudah berhenti** (satu folder run di NAS; dua proses = rusak).
2. `git clone https://github.com/fakmalpradana/cascade-sam3` (atau `git pull` bila sudah ada).
3. Python 3.12 (`py install 3.12` bila belum), lalu `scripts\windows\setup_env.ps1`. Perbaiki minimal bila gagal.
4. User sendiri: `.venv\Scripts\hf.exe auth login` (**jangan ketik token**). Perlu untuk SAM 3 (Fase 5).
5. Buat `configs/local.yaml` (isi di bawah). Cek `W:`/`Y:` terlihat; bila Claude Code jalan sebagai Admin,
   drive mapping tak terlihat → pakai path UNC (sudah dipakai di bawah).
6. `scripts\windows\00_check_env.ps1` (`check-env --smoke`). GDAL/ECW (QGIS) hanya perlu bila mengulang
   convert-ecw; boleh dilewati bila hanya item GDAL yang gagal.
7. Lanjut Fase 4:
   - Bila ada `checkpoint_<n>.ckpt` → `cascade-sam3 train-detector --resume` (setelan batch/accum **sama**).
   - Bila tidak ada → hapus `events.out.tfevents.*` & `metrics.csv` di folder run, lalu `train-detector`.
   - Jalankan dengan log ke file, mis. (Git Bash):
     `PYTHONUNBUFFERED=1 .venv/Scripts/cascade-sam3.exe train-detector > "<work_dir>/logs/train_detector_console.log" 2>&1`
   - Cek 5 menit pertama: it/s, `max_mem`, dan "Shared Usage" GPU (Get-Counter
     `'\GPU Process Memory(*)\Shared Usage'`) harus kecil (< 1 GB). Bila meluber → turunkan `max_vram_fraction`.
8. Tiap checkpoint/epoch: `predict --mode detector --name detector` + `validate --name detector`; bandingkan
   dengan `reports/validation_sam3_zeroshot.md`. Laporkan ke user tiap epoch; user ingin hasil cepat —
   bila F1 ≥ 0,740 dan stabil, tawarkan stop lebih awal.
9. Fase 5: `train-sam3`, lalu `predict --mode cascade` + `validate`, catat di README.

### configs/local.yaml untuk PC baru (gitignored)

Salin persis (path UNC; `local.example.yaml` masih memakai `Y:`/`D:`):

```yaml
work_dir: "//server_12/Server_12/2026/25 RND/cascade-sam3-work"
data:
  gdal_bin: null
  target_gsd_m: 0.08          # disetujui user 2026-11-04
cities:
  dki:
    ortho: "//Server_11/DATA_Server11/01 DKI JAKARTA 2025/02 DIGITASI/00 DATA DASAR/01 ORTHOFOTO"
    labels: "//Server_11/DATA_Server11/01 DKI JAKARTA 2025/02 DIGITASI/03 DATA FIX/03 FINAL"
    label_name_regex: '[-_]BO(_TAMBAL|_\d+)?\.gpkg$'   # hanya BO (disetujui user)
    coverage_mode: file_bbox
    coverage_buffer_m: 30
    coverage_polygon: null
    label_filter: null
  bontang:
    ortho: "//Server_11/DATA_Server11/2025/05 BONTANG/0_ORTO TAHAP 1 BONTANG"
    labels: "//Server_11/DATA_Server11/2025/05 BONTANG/24_DELIVERABLE PRODUCT/00 Kegiatan Pemodelan 3D Bangunan dan Infrastruktur serta Validasi Data PBB/02 PEMBUATAN MODEL 3D (3D CITY MODEL)/01 TAPAK BANGUNAN/01 FORMAT SHAPEFILE"
    label_name_regex: null
    coverage_mode: buffer
    coverage_buffer_m: 30
    coverage_polygon: null
    label_filter: null        # semua TOPONIMI = "Bangunan"
detector:
  batch_size: 1              # samakan dgn run PC lama agar --resume konsisten
  grad_accum_steps: 16
  max_train_images: 6000     # permintaan user: subset 6.000 tile (valid/test utuh)
  checkpoint_interval: 1
  extra: {progress_bar: tqdm}
  max_vram_fraction: 0.9     # PC baru tanpa aplikasi lain; tetap dibatasi agar tidak meluber ke RAM
sam3:
  max_vram_fraction: 0.9
```

## Cara kerja (aturan user)

- Satu masalah per langkah. Jangan lanjut fase sebelum "Lulus jika" terpenuhi.
- Tiap perbaikan kode: `pytest` + `ruff check src tests`, commit (trailer Co-Authored-By/Claude-Session) + push `main`.
- Perubahan local.yaml untuk label_filter / coverage_mode / target_gsd_m tunjukkan ke user dulu.
- Jangan commit data, bobot, raster, `configs/local.yaml`. Jangan setel ambang pakai area uji (pakai valid).
- Jangan ketik password/token.

## Jebakan teknis di PC ini

- Git Bash heredoc/`sed` memakan backslash (`\t` jadi tab) → tulis skrip Python lewat tool Write.
- PowerShell 5.1: here-string + `git commit -m` rusak → `git commit -F -` dari bash heredoc.
- `gh` tidak terpasang; cek CI via `curl https://api.github.com/repos/fakmalpradana/cascade-sam3/actions/runs?per_page=1`.
- Push kadang gagal 500 dari GitHub → ulangi.
- `log` file di NAS: mtime tidak andal; baca isinya (`tail -c ... | tr '\r' '\n'`).
