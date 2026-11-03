# cascade-sam3 — Rencana Implementasi Detail (untuk Claude Code)

> Taruh file ini di root repo sebagai `PLAN.md`. Claude Code: baca seluruh file ini sebelum menulis kode,
> kerjakan per fase, dan jangan lanjut ke fase berikutnya sebelum kriteria lulus fase terpenuhi.

Tanggal: 3 Nov 2026 · Pemilik: Fairuz Akmal Pradana (`fakmalpradana`) · Repo: `fakmalpradana/cascade-sam3` (public)

---

## 0. Ringkasan

Membangun model ekstraksi tapak bangunan dari orthophoto untuk menggantikan SAM-YOLO, dengan arsitektur
**cascade hybrid**:

```
Tile ortho 1008 px
 ├─ RF-DETR-Seg (fine-tuned, Apache-2.0) ──► kotak + mask kasar + skor
 └─ SAM 3 prompt teks "building" (zero-shot) ─► mask konsep (penambah recall)
        │
        ▼ pencocokan instance (mask IoU) → gabung, buang duplikat
        ▼ SAM 3 prompt kotak per bangunan (mask decoder di-fine-tune) → tepi atap tajam
        ▼ mask NMS → resolusi overlap → poligonisasi → regularisasi → GPKG
```

Target: mengalahkan baseline **SAM 3 zero-shot** (dan cascade > detektor saja) pada **6 area uji yang sama**
(3 DKI + 3 Bontang), diukur dengan skrip validasi
yang sama untuk semua model.

Lingkungan training: Windows, Intel i7, **RTX 3060 12 GB**, RAM 64 GB. Kode dikembangkan & dijalankan oleh Claude
Code langsung di PC tersebut.

---

## 1. Data (sudah diketahui)

| Kota | Ortho | Digitasi |
|---|---|---|
| DKI Jakarta | `Y:\01 DKI JAKARTA 2025\02 DIGITASI\00 DATA DASAR` (ECW) | `Y:\01 DKI JAKARTA 2025\02 DIGITASI\03 DATA FIX\03 FINAL` (terpisah per grid 25 ha) |
| Bontang | `Y:\2025\05 BONTANG\0_ORTO TAHAP 1 BONTANG` (ECW ±14 GB) | `Y:\2025\05 BONTANG\24_DELIVERABLE PRODUCT\00 Kegiatan Pemodelan 3D Bangunan dan Infrastruktur serta Validasi Data PBB\02 PEMBUATAN MODEL 3D (3D CITY MODEL)\01 TAPAK BANGUNAN` |

Hal yang **belum diketahui** dan harus diaudit di Fase 1 (jangan diasumsikan):
- Format digitasi (SHP / GPKG / DXF / DWG / GDB), apakah ada kelas selain bangunan di layer yang sama
  (kolom keterangan), geometri 3D (Z), garis tertutup vs poligon.
- CRS ortho & digitasi (DKI kemungkinan UTM 48S, Bontang UTM 50N — **verifikasi**), GSD ortho,
  jumlah file ECW, jumlah band (RGB/RGBA), nilai nodata/collar hitam.
- Pergeseran digitasi vs atap (tapak vs atap, sudut kamera).
- Ukuran grid DKI (diharapkan ±500 m × 500 m).

---

## 2. Fakta teknis yang sudah diverifikasi (jangan ditebak ulang)

Diverifikasi dengan `rfdetr==1.11.2`, `transformers==5.22.1`, `peft==0.21.2` (Nov 2026).

### 2.1 RF-DETR-Seg
- Kelas: `rfdetr.RFDETRSegNano/Small/Medium/Large/XLarge/2XLarge` — **semua ukuran Seg berlisensi Apache-2.0**.
  (Hindari RF-DETR *deteksi* XL/2XL: lisensi PML 1.0.)
- `RFDETRSegLarge`: resolusi default 504, `patch_size=12`, `num_windows=2` → resolusi harus kelipatan **24**;
  `num_queries=200` (maks instance per tile — pilih ukuran tile ground agar bangunan per tile < ~150).
- Training: `model.train(**kwargs)`; kwargs yang relevan: `dataset_dir, output_dir, epochs, batch_size,
  grad_accum_steps, lr (1e-4), lr_encoder (1.5e-4), num_workers, resolution, device, early_stopping,
  early_stopping_patience, checkpoint_interval, tensorboard, run_test, seed, resume, eval_max_dets (500),
  amp_dtype ("auto")`. Butuh `pip install "rfdetr[train,loggers]"`.
- Format dataset (`dataset_file="roboflow"`): `dataset_dir/{train,valid,test}/_annotations.coco.json`
  + gambar di folder yang sama. **Folder `test/` wajib ada.** Pakai satu kategori `{"id":1,"name":"building"}`.
- Checkpoint keluaran: `checkpoint_best_total.pth` (juga `checkpoint_best_ema.pth`, `last.ckpt`).
  Load: `RFDETRSegLarge.from_checkpoint(path)`; lalu `model.optimize_for_inference()` (opsional).
- Inferensi: `model.predict(rgb_numpy, threshold=0.35)` — input **RGB**; output `supervision.Detections`
  dengan `.mask` (N,H,W bool), `.confidence`, `.xyxy`.

### 2.2 SAM 3 lewat Hugging Face Transformers (bukan repo Meta)
- Alasan: repo resmi Meta butuh Triton/FlashAttention yang rawan di Windows; implementasi Transformers tidak.
- Checkpoint `facebook/sam3` **gated**: ajukan akses di halaman model, lalu `hf auth login` di PC Windows.
- Transformers v5 memakai argumen `dtype=` (bukan `torch_dtype=`).
- **Prompt kotak (PVS):** `Sam3TrackerModel`, `Sam3TrackerProcessor`, image processor `Sam3ImageProcessor`
  (resize ke 1008×1008, mask low-res 288×288).
  - `emb = model.get_image_embeddings(pixel_values)` → list 3 level:
    `[1,32,288,288]`, `[1,64,144,144]`, `[1,256,72,72]` (±10,6 MB/tile dalam fp16).
  - `out = model(image_embeddings=emb, input_boxes=boxes[None], multimask_output=False)` →
    `out.pred_masks` `(B, N, 1, 288, 288)` logits, `out.iou_scores` `(B, N, 1)`.
  - `processor(images=rgb, input_boxes=[[[x1,y1,x2,y2], ...]], return_tensors="pt")` otomatis menskalakan kotak ke
    ruang 1008; `processor.post_process_masks(pred_masks, inputs["original_sizes"])` → list `(N,1,H,W)` bool.
  - Modul: `vision_encoder` (besar, dibekukan), `prompt_encoder`, `mask_decoder` (**±4,2 jt parameter**).
- **Prompt teks (PCS):** `Sam3Model` + `Sam3Processor`:
  `vision = model.get_vision_features(pixel_values=...)`, `out = model(vision_embeds=vision, input_ids=..., attention_mask=...)`,
  `processor.post_process_instance_segmentation(out, threshold, mask_threshold=0.5, target_sizes=[(h,w)])[0]`
  → `masks, boxes, scores`. Cache embedding teks per prompt.
- Tes CPU tanpa unduh bobot: bangun model mini dengan `Sam3TrackerConfig()` lalu kecilkan
  `vision_config.backbone_config` (`hidden_size=64, intermediate_size=128, num_hidden_layers=2,
  num_attention_heads=2, global_attn_indexes=[1]`) — sudah terbukti jalan (6,8 jt parameter).

### 2.3 ECW
- Wheel `rasterio` di PyPI **tidak punya driver ECW**. QGIS (OSGeo4W) punya.
- Solusi: panggil `gdalinfo` / `gdal_translate` / `gdaladdo` milik QGIS lewat
  `cmd /c call "<QGIS>\bin\o4w_env.bat" && gdal_translate ...`, auto-deteksi di
  `C:\Program Files\QGIS*\bin`, `C:\OSGeo4W*\bin`, atau `data.gdal_bin` di config.
- Konversi hanya area berlabel (+buffer 200 m) via `-projwin`, ke GeoTIFF tiled 512, `COMPRESS=JPEG`,
  `PHOTOMETRIC=YCBCR`, `JPEG_QUALITY=92`, `BIGTIFF=YES`, lalu overview `2 4 8 16 32 64`.
- Hasil konversi disimpan di `work_dir/ortho_converted/<kota>/<nama>.tif`.

### 2.4 Keputusan yang diubah dari rencana awal
- **Cache embedding SAM 3 TIDAK dipakai default**: ±10,6 MB/tile → 20 ribu tile ≈ 200 GB. Gantinya: encoder dibekukan
  dan dijalankan di bawah `torch.no_grad()` + bf16 setiap langkah, dengan **hingga 24 kotak per gambar per langkah**
  agar biaya encoder teramortisasi. Ini muat di 12 GB.
- **Fine-tune seluruh mask decoder** sebagai default (hanya ±4,2 jt parameter); LoRA (`peft.inject_adapter_in_model`
  pada semua `nn.Linear` di `mask_decoder`) tersedia sebagai opsi `sam3.train_mode: lora`.

---

## 3. Lisensi (wajib dicantumkan di repo)

- Kode repo: Apache-2.0 (`LICENSE`).
- `NOTICE`: RF-DETR (Apache-2.0); SAM 3 di bawah **SAM License** Meta (komersial boleh, royalty-free; salinan lisensi
  harus ikut saat model/turunan dibagikan; dilarang untuk militer/perang, nuklir, spionase, ITAR, pihak tersanksi;
  Meta dapat mengubah syarat). Bobot SAM 3 & bobot fine-tune **tidak** di-commit ke repo.
- Jangan pakai fork Mask2Former berlisensi CC BY-NC.

---

## 4. Struktur repo

```
cascade-sam3/
├─ PLAN.md, README.md (Bahasa Indonesia), CLAUDE.md, LICENSE, NOTICE, .gitignore
├─ pyproject.toml              # package `cascade_sam3`, entry point `cascade-sam3`
├─ configs/local.example.yaml  # path data user; disalin ke configs/local.yaml (gitignored)
├─ scripts/windows/
│   ├─ setup_env.ps1           # venv Python 3.12, torch cu128, pip install -e .[train,dev], hf login
│   ├─ 00_check_env.ps1
│   ├─ 01_prepare_data.ps1     # inspect → convert-ecw → prepare
│   ├─ 02_train.ps1            # train-detector → train-sam3
│   ├─ 03_test_inference.ps1   # predict mode cascade + detector + concept (ablation)
│   └─ 04_validate.ps1         # validate semua run + baseline SAM-YOLO
├─ src/cascade_sam3/
│   ├─ default.yaml            # semua parameter (lihat §5)
│   ├─ config.py               # load default + local.yaml + --set key=val
│   ├─ cli.py                  # subcommand (lihat §6)
│   ├─ env_check.py
│   ├─ utils/{geo.py, log.py}
│   ├─ data/{sources.py, ecw.py, catalog.py, inspect_data.py, testsets.py, build.py}
│   ├─ models/{instances.py, detector.py, sam3.py, fusion.py, cascade.py}
│   ├─ train/{train_detector.py, train_sam3.py}
│   ├─ infer/predict.py
│   ├─ post/polygon.py
│   └─ eval/{metrics.py, validate.py}
├─ tests/                      # data sintetis, tanpa GPU & tanpa bobot
└─ .github/workflows/ci.yml
```

`.gitignore`: `configs/local.yaml`, `work/`, `*.pth`, `*.pt`, `*.ckpt`, `*.tif`, `*.ecw`, `.venv/`.

Isi `work_dir` (default `D:/cascade-sam3-work`, butuh >150 GB kosong):
```
ortho_converted/<kota>/*.tif
catalog/<kota>/{catalog.json, labels.gpkg, coverage.gpkg}
testsets/test_areas.gpkg (1 layer per kota)
testsets/<kota>_<nn>/{image.tif, gt.gpkg, area.gpkg, quicklook.png, meta.json}
dataset/coco/{train,valid,test}/ + _annotations.coco.json; dataset/{tiles.csv, stats.json}
runs/detector/, runs/sam3/
predictions/<run>/<area>.gpkg (+ .json waktu proses)
reports/{data_report.md, validation_<run>.{md,json,csv}, errors_<run>/errors_<area>.gpkg}
logs/<cmd>_<timestamp>.log
```

---

## 5. Konfigurasi (`default.yaml`) — nilai awal

```yaml
work_dir: ./work
data:
  raster_exts: [.tif, .tiff, .ecw, .jp2, .vrt, .img]
  vector_exts: [.shp, .gpkg, .geojson, .json, .dxf, .kml]   # + folder .gdb
  gdal_bin: null
  convert_clip_buffer_m: 200
  target_gsd_m: null          # null = GSD asli; set setelah inspect (mis. 0.10) bila GSD < 8 cm
  tile_size: 1008
  tile_overlap: 0.25
  min_tile_coverage: 0.9
  max_nodata_fraction: 0.05
  min_building_area_m2: 4.0
  keep_empty_tile_ratio: 0.1
  max_tiles_per_city: 10000
  image_format: jpg
  jpeg_quality: 95
  val_fraction: 0.15
  split_block_m: 500
  seed: 42
testsets:
  areas_per_city: 3
  area_size_m: 500            # 25 ha, sama dengan grid DKI
  min_buildings: 30
  min_separation_m: 1500
  exclusion_buffer_m: 50
  context_margin_m: 40
  manual_areas: null          # GPKG opsional (kolom city, name) untuk memilih area manual
detector:
  variant: seg-large
  resolution: 504
  epochs: 60
  batch_size: 2
  grad_accum_steps: 8
  lr: 1.0e-4
  lr_encoder: 1.5e-4
  num_workers: 4
  early_stopping: true
  early_stopping_patience: 12
  checkpoint_interval: 5
  extra: {}
sam3:
  model_id: facebook/sam3
  train_mode: decoder         # decoder | lora
  lora_r: 16
  lora_alpha: 32
  epochs: 12
  lr: 1.0e-4
  weight_decay: 0.01
  max_instances_per_image: 24
  box_jitter: 0.1
  mask_loss_size: 512
  focal_weight: 20.0
  dice_weight: 1.0
  iou_weight: 1.0
  num_workers: 2
  amp_dtype: bfloat16
  max_train_images: null
  max_val_images: 300
infer:
  mode: cascade               # cascade | detector | concept
  tile_size: 1008
  overlap_px: 256
  det_threshold: 0.35
  use_concept: true
  concept_prompts: ["building"]
  concept_threshold: 0.5
  concept_add_threshold: 0.6
  concept_score_weight: 0.8
  match_iou: 0.3
  use_refine: true
  refine_min_iou_pred: 0.70
  refine_consistency_iou: 0.5
  batch_boxes: 32
  nms_iou: 0.5
  nms_containment: 0.8
  min_area_m2: 4.0
  simplify_tolerance_m: 0.15
  regularize: true
  rect_threshold: 0.85
eval:
  iou_thresholds: [0.25, 0.5, 0.75]
  primary_iou: 0.5
  boundary_tolerance_m: 0.5
  pixel_gsd_m: null
  min_area_m2: 2.0
  score_sweep: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
```

`configs/local.example.yaml` berisi `work_dir: "D:/cascade-sam3-work"` dan blok `cities.dki` / `cities.bontang`
dengan path di §1 (gunakan `/`), plus `coverage_mode` (`file_bbox` untuk DKI karena per grid; `buffer` 30 m untuk
Bontang), `coverage_polygon: null`, `label_filter: null` (diisi setelah inspect jika layer bercampur kelas,
contoh `{column: "KETERANGAN", values: ["BANGUNAN"]}`).

---

## 6. CLI (`cascade-sam3 [--config F] [--set k=v ...] <cmd>`)

| Perintah | Fungsi |
|---|---|
| `check-env [--smoke]` | torch+CUDA, VRAM ≥ 11 GB, bf16, paket, GDAL QGIS + driver ECW, akses HF `facebook/sam3`, path data, ruang disk; `--smoke` menjalankan SAM 3 sekali di GPU & mencetak VRAM puncak |
| `inspect` | `reports/data_report.md/.json` (lihat §7.1) |
| `convert-ecw [--city] [--full] [--force]` | ECW → GeoTIFF (§2.3) |
| `catalog [--city]` | indeks raster, CRS kerja, label harmonis, coverage |
| `make-testsets [--city]` | pilih & ekspor 3 area uji per kota |
| `build-dataset [--city]` | tile + COCO |
| `prepare` | catalog + make-testsets + build-dataset |
| `train-detector [--name] [--resume] [--epochs]` | RF-DETR-Seg |
| `train-sam3 [--name] [--resume] [--epochs]` | fine-tune decoder SAM 3 |
| `predict [--mode] [--detector] [--sam3] [--zero-shot-sam3] [--name] [--areas ...] [--raster F --out F]` | inferensi area uji atau raster bebas |
| `validate [--name] [--pred-dir] [--areas ...]` | metrik (§10) |

Setiap perintah menulis log ke `work_dir/logs/`. Default `--detector` = checkpoint terbaik di `runs/detector`,
default `--sam3` = `runs/sam3/sam3_decoder_best.pt`.

---

## 7. Fase dan tugas

### Fase 0 — Repo, lingkungan, CI
1. `gh repo create fakmalpradana/cascade-sam3 --public` (dari Claude Code di Windows), clone, buat struktur §4.
2. `setup_env.ps1`: cek `py -3.12`; buat `.venv`; `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`;
   `pip install -e .[train,dev]`; salin `local.example.yaml` → `local.yaml` bila belum ada; ingatkan `hf auth login`.
3. Pastikan QGIS terpasang (untuk ECW); kalau belum, minta user memasang QGIS LTR.
4. CI (`.github/workflows/ci.yml`): matrix `ubuntu-latest` + `windows-latest`, Python 3.12;
   `pip install torch --index-url https://download.pytorch.org/whl/cpu` lalu `pip install -e .[dev]`
   (tanpa extra `train` — model diimpor secara lazy); jalankan `ruff check .` dan `pytest`.
   Tes yang butuh `transformers` di-`pytest.importorskip`.

**Lulus jika:** `cascade-sam3 check-env --smoke` semua OK; CI hijau.

### Fase 1 — Audit & konversi data
`inspect` melaporkan per kota:
- Raster: nama, driver, CRS, GSD, ukuran px, band, ukuran GB, terbaca rasterio atau tidak
  (fallback metadata lewat `gdalinfo -json` QGIS), fraksi overlap dengan bbox label.
- Vektor: tiap file/layer, tipe geometri, CRS, kolom, nilai unik kolom teks (≤25 nilai).
- Label: jumlah poligon, persentil luas p5/p50/p95, jumlah < 4 m², bbox, ukuran ekstent file pertama.

Lalu `convert-ecw` (hanya area berlabel). Setelah itu Claude Code menyesuaikan `local.yaml`
(`label_filter`, `coverage_mode`, `target_gsd_m`) berdasarkan laporan dan **menunjukkan ringkasannya ke user**.

Aturan pemuatan label (`sources.py`):
- Cari rekursif; abaikan sidecar SHP; dukung multi-layer (pyogrio `list_layers`) & folder `.gdb`.
- `force_2d`; garis tertutup → poligon; `make_valid`; pecah multipart; buang luas < `min_building_area_m2`.
- Reproyeksi ke CRS kerja; hapus duplikat di sambungan grid (IoU ≥ 0,9 via STRtree).
- Simpan `src_file`, `building_id`.

Coverage berlabel (area di mana "tidak ada label" berarti "bukan bangunan"):
- `file_bbox`: gabungan bbox tiap file (DKI); `buffer`: union buffer 30 m tiap bangunan lalu `buffer(-10)`;
  `polygon`: file batas manual. Selalu diiris dengan footprint raster.

**Lulus jika:** laporan data lengkap, konversi selesai, user mengonfirmasi `label_filter`/CRS benar.

### Fase 2 — Katalog, area uji, dataset
**Katalog:** pilih CRS kerja = CRS terbanyak di antara raster terbaca; raster ber-CRS lain dilewati & dilaporkan.
Simpan `catalog.json` (raster + stretch persentil 1–99 bila bukan uint8), `labels.gpkg`, `coverage.gpkg`.

**Area uji (3 per kota, `testsets.py`):**
1. Kandidat: jika label per grid berukuran ~`area_size_m` (median sisi 0,6–1,4×) → kotak 500 m berpusat di tiap grid;
   selain itu grid 500 m dengan langkah 250 m di atas coverage.
2. Syarat kandidat: ≥ 98% di dalam coverage; area + margin konteks 40 m termuat penuh di satu raster;
   ≥ `min_buildings` bangunan (centroid di dalam); fraksi piksel valid ≥ 95% (baca thumbnail 96×96).
3. Hitung kepadatan = luas atap / luas area. Bagi kandidat ke 3 kelompok kuantil (rendah/sedang/tinggi).
4. Pilih 1 per kelompok (mulai dari kelompok tengah): kandidat terdekat ke median kelompok yang berjarak
   ≥ `min_separation_m` dari area terpilih (longgarkan ke ½ lalu 0 bila tak ada). Deterministik (`seed`).
5. Ekspor `testsets/<kota>_<nn>/`: `image.tif` (area + margin, RGB JPEG/YCbCr tiled + internal mask),
   `gt.gpkg` (bangunan yang beririsan area+margin), `area.gpkg`, `quicklook.png` (outline GT kuning, area magenta),
   `meta.json` (raster sumber, GSD, jumlah GT, kelas kepadatan).
6. Opsi `testsets.manual_areas` untuk menimpa pilihan otomatis.

**Dataset (`build.py`):**
- Tile persegi `tile_size × GSD` meter, overlap 25%, harus di dalam satu raster & ≥ 90% di dalam coverage.
- Tile yang centroid-nya di area uji → split `test` (hanya evaluasi RF-DETR); tile yang menyentuh area uji
  + (`exclusion_buffer_m` + `context_margin_m`) → **dibuang** (cegah bocor).
- Sisanya: split spasial per blok 500 m (hash md5 blok + seed) → `valid` 15%, `train` 85%.
- Subsampel acak hingga `max_tiles_per_city`; tile kosong disimpan dengan peluang 10%; tile dengan nodata > 5% dibuang.
- Anotasi COCO: poligon terpotong ke tile, koordinat piksel, bbox, area, `iscrowd=0`, plus `building_id` & `truncated`
  (field tambahan). Lubang poligon diabaikan (catat sebagai keterbatasan).
- Tulis `tiles.csv` & `stats.json` (tile & bangunan per split per kota). Buat 12 overlay acak per kota untuk cek
  kesejajaran label vs atap; tunjukkan ke user.

**Lulus jika:** 6 area uji tersebar (3 kepadatan per kota), quicklook terlihat pas, `valid` tidak kosong,
tidak ada tile train/valid yang beririsan area uji (tulis tes untuk ini).

### Fase 3 — Baseline
> **Diubah 4 Nov 2026 (keputusan user, opsi C):** bobot SAM-YOLO tidak tersedia dan hasil lamanya tidak cukup
> sebagai baseline. Acuan utama = **SAM 3 zero-shot**; pembanding internal = RF-DETR detector-only (Fase 4).
> Bila prediksi SAM-YOLO tersedia kelak, tambahkan saja: `validate --pred-dir <folder> --name sam_yolo`.

1. `predict --mode concept --zero-shot-sam3 --name sam3_zeroshot` lalu `validate --name sam3_zeroshot`.

**Lulus jika:** laporan baseline `sam3_zeroshot` tersimpan; angka acuan dicatat di README.

### Fase 4 — Training detektor
- `train-detector`: `RFDETRSegLarge` dengan kwargs §2.1 dari config; `run_test=True`; simpan argumen ke
  `runs/detector/cascade_train_args.json`. Pantau dengan TensorBoard.
- Jika OOM: turunkan `resolution` ke 432 atau `batch_size` 1 + `grad_accum_steps` 16; bila masih OOM, `seg-medium`.
- Estimasi waktu ukur dari epoch pertama; laporkan ke user sebelum lanjut penuh.

**Lulus jika:** `predict --mode detector` + `validate` memberi F1 objek ≥ baseline `sam3_zeroshot`, atau bila belum,
analisis error layer dan sesuaikan (GSD, tile, epoch) sebelum lanjut.

### Fase 5 — Fine-tune SAM 3 (mask decoder)
Dataset `BoxPromptDataset` dari COCO yang sama:
- Per gambar: hingga 24 anotasi acak; mask GT di-`cv2.fillPoly` ukuran tile; augmentasi dihedral (rot90 × flip)
  hanya untuk train; kotak = bbox mask + jitter Gaussian (σ = 0,1 × ukuran kotak; val 0,05, deterministik).
- `pixel_values` via `Sam3ImageProcessor`; kotak diskalakan ke ruang 1008; GT diresize ke 512 (INTER_AREA > 0,5).

Loop:
- Load `Sam3TrackerModel` fp32 → bekukan semua → trainable = `mask_decoder` (atau LoRA) → `vision_encoder` di-cast
  ke bf16 → `.to("cuda")`.
- Tiap langkah: `with torch.no_grad(), autocast(bf16): emb = get_image_embeddings(...)`;
  lalu decoder di bawah autocast dengan `input_boxes`, `multimask_output=False`;
  logits 288 → interpolasi bilinear ke 512.
- Loss = 20 × focal (α 0,25, γ 2) + 1 × dice + 1 × MSE(iou_scores, IoU nyata).
- AdamW lr 1e-4, wd 0,01, warmup ≤ 500 langkah + cosine; clip grad 1,0.
- Evaluasi mIoU val **sebelum training (zero-shot)** dan tiap epoch; simpan `sam3_decoder_best.pt` &
  `sam3_decoder_last.pt` berisi `{mode, lora_r, lora_alpha, state_dict (mask_decoder), optimizer, scheduler, epoch,
  val_iou, best, model_id}`; dukung `--resume`. Log CSV + TensorBoard. Catat VRAM puncak.

**Lulus jika:** mIoU val naik dari zero-shot, dan pada area uji `cascade` > `detector` dalam boundary F1 / IoU piksel.

### Fase 6 — Inferensi, fusi, ablation
Per tile (`cascade.py`):
1. Detektor (`det_threshold`) → Instances (mask, skor, sumber `det`).
2. SAM 3 teks untuk tiap prompt (`concept_threshold`) → Instances `concept`.
3. Fusi (`fusion.py`): IoU mask (grid stride 4). Det yang cocok (IoU ≥ 0,3) dengan konsep → skor
   `s_d + (1 − s_d) × 0,8 × s_c`, sumber `det+concept`. Konsep tanpa pasangan dengan skor ≥ 0,6 ditambahkan
   (skor × 0,8).
4. Refinement: kotak dari mask + padding 3% (min 2 px) → `BoxRefiner` (embedding sekali per tile, kotak per batch 32).
   Ganti mask bila `iou_pred ≥ 0,70` dan IoU(mask baru, mask lama) ≥ 0,5.
5. Mask NMS (IoU > 0,5 atau containment > 0,8), potong ke piksel valid, lalu tiap piksel milik instance skor
   tertinggi; buang < `min_area`.

Raster penuh (`predict.py`): sliding window 1008 px, overlap 256 px, di GSD yang sama dengan training.
- Tiap instance → poligon terbesar (`rasterio.features.shapes`) → regularisasi: jika luas/luas min-rotated-rect ≥ 0,85
  dan tanpa lubang → pakai persegi panjangnya; selain itu simplify 0,15 m.
- Tandai `partial` bila mask menyentuh tepi tile yang bukan tepi wilayah.
- Gabung antar tile: poligon utuh di-NMS (IoU > 0,3 atau containment > 0,6); poligon parsial yang > 50% tertutup
  poligon utuh dibuang, sisanya dikelompokkan (union-find, irisan > 5% luas terkecil) → union → regularisasi.
- Output GPKG layer `buildings`: `score, source, refined, partial, area_m2` + JSON waktu proses.

Ablation wajib (run name terpisah): `detector`, `concept`, `cascade` tanpa konsep (`--set infer.use_concept=false`),
`cascade` penuh. Tuning ambang dilakukan pada **valid**, bukan area uji.

### Fase 7 — Paket & serah terima
README final (langkah Windows dari nol, tabel hasil semua run), ekspor ONNX detektor (opsional), skrip
`predict --raster` untuk dipakai NARAGA, tag rilis `v0.1.0` (tanpa bobot di repo).

---

## 8. Skrip Windows

- `00_check_env.ps1`: aktifkan `.venv`, `cascade-sam3 check-env --smoke`.
- `01_prepare_data.ps1`: `inspect` → jeda konfirmasi → `convert-ecw` → `prepare`.
- `02_train.ps1 [-SkipDetector] [-SkipSam3]`: `train-detector` lalu `train-sam3`.
- `03_test_inference.ps1`: `predict --mode cascade`, `--mode detector`, `--mode concept`.
- `04_validate.ps1 [-BaselineDir <folder SAM-YOLO>]`: `validate` untuk semua run + baseline, lalu cetak tabel ringkas.
- Semua skrip `$ErrorActionPreference = "Stop"`, berhenti bila exit code ≠ 0, dan menulis transcript ke `work_dir/logs`.

---

## 9. CLAUDE.md (isi untuk repo)

- Jalankan perintah lewat `.venv\Scripts\cascade-sam3.exe` atau `python -m cascade_sam3`.
- Jangan commit data, bobot, atau `configs/local.yaml`.
- Sebelum mengubah perilaku model, jalankan `pytest`; setelah perubahan, jalankan ulang `validate` dan bandingkan
  dengan run sebelumnya.
- Untuk error data, baca `work_dir/reports/data_report.md` dan log terbaru di `work_dir/logs/`.
- Jangan menyetel ambang memakai area uji; gunakan split `valid`.

---

## 10. Validasi (`validate`) — definisi metrik

Untuk tiap area uji: GT dan prediksi dipotong ke poligon area, buang < 2 m².

**Piksel** (rasterisasi pada GSD `image.tif`, hanya di dalam poligon area): TP/FP/FN/TN →
Overall Accuracy, Precision, Recall, **F1**, **IoU bangunan**, IoU latar, mIoU, Kappa Cohen.

**Objek** (per bangunan): matriks IoU via STRtree → pencocokan satu-ke-satu **Hungarian**
(`scipy.optimize.linear_sum_assignment`) → pada IoU 0,25 / **0,5 (utama)** / 0,75: TP, FP, FN, Precision
(correctness), Recall (completeness), **F1**, rerata IoU pasangan.

**AP**: gaya COCO, interpolasi 101 titik, pencocokan greedy berurutan skor → AP50, AP75, AP50:95.

**Boundary F1**: panjang batas prediksi dalam buffer 0,5 m dari batas GT (precision) dan sebaliknya (recall).

**Luas & jumlah**: total luas GT vs prediksi, galat relatif, selisih jumlah.

**Sweep skor**: F1 objek @0,5 untuk ambang skor 0,3–0,8 → rekomendasi ambang.

Agregasi: per kota & keseluruhan (jumlahkan confusion piksel dan TP/FP/FN objek; AP dan boundary dirata-rata).
Output: `validation_<run>.md` (tabel), `.json`, `.csv`, dan `errors_<run>/errors_<area>.gpkg`
(layer `gt` berstatus TP/FN, layer `pred` TP/FP + IoU + skor) untuk ditinjau di QGIS.
Prediksi yang hilang untuk suatu area dihitung sebagai kosong (bukan error).

---

## 11. Tes (CI, tanpa GPU)

- **Kota sintetis** (fixture): GeoTIFF RGB EPSG:32748 3000×3000 px @0,1 m dengan atap terang berotasi; label dipecah
  ke 9 file SHP grid 100 m dengan kolom `KETERANGAN`.
- `test_metrics`: skor piksel kasus diketahui; Hungarian menghitung duplikat sebagai FP; AP = 1 untuk urutan benar
  dan lebih kecil untuk urutan terbalik; boundary F1 = 1 / 0.
- `test_fusion`: penambahan konsep, pencocokan, refinement ditolak saat `iou_pred` rendah, NMS, resolusi overlap.
- `test_pipeline`: `catalog → make-testsets → build-dataset` pada kota sintetis (area 60 m, tile 256); cek 3 area,
  COCO valid, **tidak ada tile train/valid yang beririsan zona eksklusi**; lalu `predict_raster` dengan prediktor palsu
  (threshold kecerahan + connected components) → `validate` harus memberi F1 objek > 0,9.
- `test_stitching`: bangunan yang terpotong 2–4 tile menjadi satu poligon.
- `test_sam3_train` (skip bila `transformers` tak terpasang): model mini (§2.2) 1 epoch di CPU pada dataset sintetis,
  checkpoint tersimpan dan bisa dimuat ulang oleh `load_decoder_weights`.
- `test_cli`: parser semua subcommand; override `--set`.

---

## 12. Risiko & mitigasi

| Risiko | Mitigasi |
|---|---|
| OOM 12 GB | encoder SAM 3 no_grad + bf16, maks 24 kotak/langkah; RF-DETR resolusi/batch turun |
| Akses `facebook/sam3` belum disetujui | ajukan sekarang; Fase 1–4 tidak butuh SAM 3 |
| QGIS tanpa driver ECW | pakai QGIS LTR standar (OSGeo4W); `check-env` memverifikasi `gdalinfo --formats` |
| Digitasi bergeser dari atap | overlay acak Fase 2; bila perlu koreksi offset per area |
| Kampung padat → mask tergabung | detektor instance di depan, prompt kotak per bangunan, evaluasi per kelas kepadatan |
| Tile terlalu banyak → epoch lama | `max_tiles_per_city`, `target_gsd_m` |
| Bocor data uji | eksklusi buffer + tes otomatis |
| Syarat SAM License berubah | simpan salinan lisensi saat unduh; mode `detector` tetap berdiri sendiri |

---

## 13. Checklist untuk user (sebelum Fase 0)

- [ ] Ajukan akses Hugging Face ke `facebook/sam3`.
- [ ] Pastikan QGIS terpasang di PC Windows.
- [ ] Siapkan drive kerja dengan > 150 GB kosong (default `D:/cascade-sam3-work`).
- [x] ~~Siapkan bobot/skrip SAM-YOLO lama untuk baseline Fase 3.~~ Tidak tersedia; baseline = SAM 3 zero-shot (opsi C).
- [ ] Tentukan format output akhir (GPKG default) dan CRS target bila berbeda dari CRS ortho.
