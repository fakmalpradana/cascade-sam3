# cascade-sam3

Ekstraksi tapak bangunan dari orthophoto dengan arsitektur **cascade hybrid**, pengganti SAM-YOLO:

```
Tile ortho 1008 px
 ├─ RF-DETR-Seg (fine-tuned, Apache-2.0) ──► kotak + mask kasar + skor
 └─ SAM 3 prompt teks "building" (zero-shot) ─► mask konsep (penambah recall)
        ▼ pencocokan instance (mask IoU) → gabung, buang duplikat
        ▼ SAM 3 prompt kotak per bangunan (mask decoder di-fine-tune) → tepi atap tajam
        ▼ mask NMS → resolusi overlap → poligonisasi → regularisasi → GPKG
```

Rencana detail & definisi metrik: [`PLAN.md`](PLAN.md). Lisensi: Apache-2.0 (kode), lihat [`NOTICE`](NOTICE)
untuk RF-DETR dan SAM 3 (SAM License Meta). Bobot model **tidak** ada di repo.

## Kebutuhan

- Windows 10/11, GPU NVIDIA ≥ 12 GB (diuji: RTX 3060 12 GB), RAM 64 GB.
- Python 3.12 (python.org, dengan *py launcher*), Git, QGIS LTR (untuk driver ECW).
- Drive kerja dengan > 150 GB kosong (default `D:/cascade-sam3-work`).
- Akses Hugging Face ke [`facebook/sam3`](https://huggingface.co/facebook/sam3) (gated; ajukan dulu).

## Langkah dari nol (Windows, PowerShell)

```powershell
git clone https://github.com/fakmalpradana/cascade-sam3.git
cd cascade-sam3
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # sekali saja, bila skrip diblokir
.\scripts\windows\setup_env.ps1                        # .venv + torch cu128 + paket
.venv\Scripts\hf.exe auth login
notepad configs\local.yaml                             # cek path data & work_dir
.\scripts\windows\00_check_env.ps1                     # Fase 0: semua harus OK
.\scripts\windows\01_prepare_data.ps1                  # Fase 1–2: inspect → convert-ecw → prepare
.\scripts\windows\02_train.ps1                         # Fase 4–5: RF-DETR lalu decoder SAM 3
.\scripts\windows\03_test_inference.ps1                # Fase 6: ablation
.\scripts\windows\04_validate.ps1                                  # semua run (opsional -BaselineDir <folder>)
```

Setiap skrip berhenti bila ada perintah gagal dan menyimpan transcript di `work_dir/logs/`.

## CLI

`cascade-sam3 [--config F] [--set k=v ...] <cmd>`

| Perintah | Fungsi |
|---|---|
| `check-env [--smoke]` | torch+CUDA, VRAM, bf16, paket, GDAL QGIS + ECW, akses HF, path data, disk |
| `inspect` | `reports/data_report.md/.json` |
| `convert-ecw [--city] [--full] [--force]` | ECW → GeoTIFF (hanya area berlabel + 200 m) |
| `catalog` / `make-testsets` / `build-dataset` / `prepare` | katalog, 3 area uji per kota, tile + COCO |
| `train-detector [--name] [--resume] [--epochs]` | RF-DETR-Seg |
| `train-sam3 [--name] [--resume] [--epochs]` | fine-tune mask decoder SAM 3 (`sam3.train_mode: lora` opsional) |
| `predict [--mode cascade\|detector\|concept] [--zero-shot-sam3] [--name] [--areas ...]` | inferensi area uji |
| `predict --raster F --out F.gpkg` | inferensi raster bebas (untuk NARAGA) |
| `regularize --src <run> --name <run_baru> [--steps v2 v3 ...]` | post-process sudut siku pada prediksi yang sudah ada |
| `validate [--name] [--pred-dir] [--areas ...]` | metrik piksel, objek, AP, boundary F1 |

Contoh override: `cascade-sam3 --set infer.use_concept=false predict --mode cascade --name cascade_noconcept`.

Baseline (Fase 3) = SAM 3 zero-shot. Bobot SAM-YOLO tidak tersedia; prediksi SAM-YOLO bisa ditambah kelak
lewat `validate --pred-dir <folder> --name sam_yolo`.

```powershell
.venv\Scripts\cascade-sam3.exe predict --mode concept --name sam3_zeroshot
.venv\Scripts\cascade-sam3.exe validate --name sam3_zeroshot
```

## Hasil

Diisi setelah Fase 3–6 (6 area uji: 3 DKI + 3 Bontang, objek @ IoU 0,5).
Baseline per kota: F1 objek DKI 0,696, Bontang 0,781 (`reports/validation_sam3_zeroshot.md`, 4 Nov 2026).

| Run | F1 objek | Precision | Recall | IoU piksel | Boundary F1 | AP50 |
|---|---|---|---|---|---|---|
| sam3_zeroshot (baseline) | 0,740 | 0,723 | 0,758 | 0,823 | 0,729 | 0,663 |
| detector (epoch 2) | 0,802 | 0,776 | 0,829 | 0,875 | 0,789 | 0,766 |
| cascade (detektor epoch 2, decoder 2 epoch) | 0,800 | 0,772 | 0,830 | 0,879 | 0,788 | 0,776 |
| tahap1_detector_default (epoch 12) | 0,811 | 0,796 | 0,827 | 0,885 | 0,800 | 0,770 |
| tahap1_detector_tuned | 0,794 | 0,756 | 0,837 | 0,885 | 0,799 | 0,776 |
| tahap1_cascade_default (decoder 6 epoch) | 0,808 | 0,789 | 0,828 | 0,886 | 0,793 | 0,774 |
| tahap1_cascade_tuned | 0,782 | 0,731 | 0,841 | 0,882 | 0,795 | 0,787 |
| tahap2_detector_default (resolusi 672) | 0,815 | 0,792 | 0,838 | 0,889 | 0,814 | 0,780 |
| tahap2_detector_tuned | 0,798 | 0,756 | 0,845 | 0,890 | 0,812 | 0,788 |
| tahap2_cascade_default | 0,813 | 0,789 | 0,839 | 0,888 | 0,795 | 0,784 |
| tahap2b_detector_default (672, semua tile) | 0,821 | 0,803 | 0,840 | 0,893 | 0,818 | 0,785 |
| tahap2b_detector_tuned | 0,804 | 0,765 | 0,846 | 0,893 | 0,817 | 0,791 |
| tahap2b_cascade_default | 0,817 | 0,796 | 0,840 | 0,889 | 0,798 | 0,786 |
| **final_detector** (= tahap2b_detector_tuned) | 0,804 | 0,765 | 0,846 | 0,893 | 0,817 | 0,791 |
| final_cascade_reg (tahap2b_cascade_default + v2 > v3 > v3) | 0,814 | 0,791 | 0,839 | 0,883 | 0,792 | 0,780 |
| samyolo (GeoJSON SAM + YOLO, evaluator sama) | 0,567 | 0,584 | 0,551 | 0,720 | 0,645 | 0,382 |

Catatan run (5 Nov 2026): `detector` = bobot `checkpoint_best_ema` epoch 2, training dihentikan lebih awal atas keputusan
user; `cascade` = decoder SAM 3 2 epoch pada 4.000 tile (mIoU val 0,846 → 0,881). Ambang masih bawaan, belum disetel
pada split `valid`. Selama/sesudah training yang dihentikan tidak ada `checkpoint_best_total.pth`, jadi `predict` perlu
`--detector <work_dir>/runs/detector/checkpoint_best_ema_ep2_stop.pth`.

Tahap 1 (5 Nov 2026): detektor 12 epoch (F1 val 0,786), decoder 6 epoch pada 4.000 tile (mIoU val 0,885). Target IoU
piksel > 0,90 **belum tercapai**. `tuned` memakai ambang dari `tune-infer` pada 300 tile `valid`
(`reports/tune_tahap1_*.json`): detektor `det_threshold=0.25`; cascade `det_threshold=0.2, concept_add_threshold=0.9,
refine_min_iou_pred=1.01` (refine mati). IoU `valid` setelah tuning ±0,90, tetapi tidak terbawa ke area uji: F1 objek
turun dan IoU naik paling banyak 0,0003. Ambang di `default.yaml` tidak diubah.

Tahap 2 (6 Nov 2026): detektor resolusi 672, warm start dari bobot epoch 12 (`detector.init_weights`), 8 epoch, 6.000
tile (`runs/detector/seg-large-672`). Target IoU piksel > 0,90 **belum tercapai**: ambang dari `valid`
(`det_threshold=0.25`, IoU `valid` 0,908) menghasilkan IoU area uji 0,8902. `cascade` tidak lebih baik dari `detector`
pada IoU (0,888 vs 0,889) maupun boundary F1 (0,795 vs 0,814). Bobot terbaru ini kini menjadi default `predict` karena
`latest_detector_ckpt` memilih berkas terbaru; pakai `--detector` untuk memilih bobot lain.

Tahap 2b (7 Nov 2026): detektor resolusi 672, warm start dari bobot Tahap 2, 8 epoch, semua 11.431 tile
(`runs/detector/seg-large-672-all`, ±12,7 jam). Target IoU piksel > 0,90 **belum tercapai**: ambang dari `valid`
(`det_threshold=0.25`, IoU `valid` 0,913) menghasilkan IoU area uji 0,8929, naik 0,0027 dari Tahap 2. F1 val hampir
datar sepanjang training (0,794 → 0,798). Area uji terlemah `dki_01` (IoU 0,857, precision objek 0,61) dan
`bontang_01` (IoU 0,871). `cascade` tetap tidak lebih baik dari `detector` pada IoU (0,889 vs 0,893) maupun boundary F1
(0,798 vs 0,818). Bobot ini kini menjadi default `predict`.

**Model final (7 Nov 2026, keputusan user: berhenti di IoU 0,893):** mode `detector`, bobot
`runs/detector/seg-large-672-all/checkpoint_best_total.pth`, `det_threshold=0.25` dari split `valid`. Target 0,90 tidak
tercapai dan `cascade` tidak dipakai karena tidak lebih baik dari `detector`. Ablation `concept`/`cascade_noconcept` tidak
dijalankan. Prediksi ulang menghasilkan metrik yang sama. Paket QGIS (TIFF + GeoJSON WGS84) ada di
`work_dir/reports/qgis_final`.

```powershell
.venv\Scripts\cascade-sam3.exe --set infer.det_threshold=0.25 predict --mode detector --detector <bobot> --name final_detector
.venv\Scripts\cascade-sam3.exe validate --name final_detector
```

Regularisasi sudut siku (7 Nov 2026): `post/rightangle.py` memuat algoritma v2 dan v3 dari user (port identik,
dicek pada 1.200 poligon nyata). Rantai default `infer.rightangle_chain: [v2, v3, v3]` dijalankan di akhir `predict`
setelah penggabungan antar tile; `[]` mematikannya. Prediksi lama di-post-process tanpa inferensi ulang:

```powershell
.venv\Scripts\cascade-sam3.exe regularize --src tahap2b_cascade_default --name final_cascade_reg
.venv\Scripts\cascade-sam3.exe validate --name final_cascade_reg
```

Terhadap label area uji, regularisasi sedikit menurunkan metrik: F1 0,817 → 0,814, IoU piksel 0,889 → 0,883, boundary
F1 0,798 → 0,792. Run sebelum 7 Nov dibuat tanpa langkah ini; untuk mengulangnya pakai
`--set "infer.rightangle_chain=[]"`. Report perbandingan dengan SAM + YOLO:
`work_dir/reports/Perbandingan_SAM-YOLO_vs_RF-DETR-SAM3.pdf`.

## Pengembangan

```powershell
.venv\Scripts\python.exe -m pytest     # data sintetis, tanpa GPU & tanpa bobot
.venv\Scripts\ruff.exe check .
```

Keterbatasan diketahui: lubang poligon diabaikan di anotasi COCO; konversi ECW memotong satu persegi
bbox label per raster.
