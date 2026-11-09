# models/

This directory is the default local location for model checkpoints
referenced by `config/app.yaml` (`models.sam21.checkpoint`,
`models.sam3.checkpoint`, `models.yolo_sam.yolo_checkpoint`).

Checkpoints are never committed to Git (see `.gitignore`). They are large
binaries distributed under their own licences, which you must review and
accept yourself before downloading. VISTLabel never downloads a model at
application startup; use `scripts/download_models.py` or fetch the files
manually per the upstream instructions.

## SAM 2.1 (`sam21`)

A Hugging Face-format snapshot *directory* (`config.json`,
`model.safetensors`, ...), loaded through the `transformers` library's
`Sam2VideoModel` / `Sam2VideoProcessor`:

```bash
python scripts/download_models.py --model sam21-base-plus
# or, for the larger/more accurate variant:
python scripts/download_models.py --model sam21-large
```

This downloads a pinned commit revision from the Hugging Face Hub into
`./models/sam2.1-hiera-<variant>/`; point `models.sam21.checkpoint` at that
directory.

## SAM 3 (`sam3`)

Text/concept search, and point/box/mask prompts with video propagation.
Runs `facebook/sam3`, a Hugging Face-format snapshot directory:

```bash
python scripts/download_models.py --model sam3
# then set models.sam3.checkpoint: ./models/sam3
```

`facebook/sam3` is **gated**: accept its licence on the model page and
expose a token (`HF_TOKEN`, or `huggingface-cli login`) first.

The one checkpoint drives two `transformers` models: `Sam3VideoModel` for
text/concept search and `Sam3TrackerVideoModel` for point, box and mask
prompts and propagation. Only one of them is in GPU memory at a time, so
the first request after switching between text search and visual prompting
is slow.

## YOLO + SAM (`yolo_sam`)

Needs a **custom** YOLO checkpoint trained on your own object classes
plus a matching `yolo_class_mapping.json`. The example
`config/yolo_class_mapping.example.json` shows the schema with a set of
laboratory classes (`pipette`, `glove`, `culture_flask`, ...). A generic
pretrained YOLO checkpoint is never loaded in its place. `models.yolo_sam.refiner_backend` picks which of
`sam21` / `sam3` turns the detected boxes into masks (`sam21` by default);
the chosen refiner must itself be enabled. Until a custom checkpoint and
class mapping are present, the backend reports itself as unavailable with
the exact reason.
