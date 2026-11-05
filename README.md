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
| concept | | | | | | |
| cascade_noconcept | | | | | | |

Catatan run (5 Nov 2026): `detector` = bobot `checkpoint_best_ema` epoch 2, training dihentikan lebih awal atas keputusan
user; `cascade` = decoder SAM 3 2 epoch pada 4.000 tile (mIoU val 0,846 → 0,881). Ambang masih bawaan, belum disetel
pada split `valid`. Selama/sesudah training yang dihentikan tidak ada `checkpoint_best_total.pth`, jadi `predict` perlu
`--detector <work_dir>/runs/detector/checkpoint_best_ema_ep2_stop.pth`.

Tahap 1 (5 Nov 2026): detektor 12 epoch (F1 val 0,786), decoder 6 epoch pada 4.000 tile (mIoU val 0,885). Target IoU
piksel > 0,90 **belum tercapai**. `tuned` memakai ambang dari `tune-infer` pada 300 tile `valid`
(`reports/tune_tahap1_*.json`): detektor `det_threshold=0.25`; cascade `det_threshold=0.2, concept_add_threshold=0.9,
refine_min_iou_pred=1.01` (refine mati). IoU `valid` setelah tuning ±0,90, tetapi tidak terbawa ke area uji: F1 objek
turun dan IoU naik paling banyak 0,0003. Ambang di `default.yaml` tidak diubah.

## Pengembangan

```powershell
.venv\Scripts\python.exe -m pytest     # data sintetis, tanpa GPU & tanpa bobot
.venv\Scripts\ruff.exe check .
```

Keterbatasan diketahui: lubang poligon diabaikan di anotasi COCO; konversi ECW memotong satu persegi
bbox label per raster.
