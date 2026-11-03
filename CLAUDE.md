# CLAUDE.md — cascade-sam3

Rencana lengkap: `PLAN.md`. Kerjakan per fase; jangan lanjut sebelum kriteria lulus fase terpenuhi.

- Jalankan perintah lewat `.venv\Scripts\cascade-sam3.exe` atau `.venv\Scripts\python.exe -m cascade_sam3`.
- Jangan commit data, bobot (`*.pth`, `*.pt`, `*.ckpt`), raster, atau `configs/local.yaml`.
- Sebelum mengubah perilaku model, jalankan `pytest`; setelah perubahan, jalankan ulang `validate`
  dan bandingkan dengan run sebelumnya (`work_dir/reports/validation_<run>.md`).
- Untuk error data, baca `work_dir/reports/data_report.md` dan log terbaru di `work_dir/logs/`.
- Jangan menyetel ambang memakai area uji; gunakan split `valid`.
- `work_dir` dibaca dari config: `cascade-sam3 show-config --key work_dir`.
- Perubahan `configs/local.yaml` (label_filter, coverage_mode, target_gsd_m) harus ditunjukkan ke user dulu.
