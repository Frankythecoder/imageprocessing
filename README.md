# HAM10000 Skin-Lesion Research Classifier

A React frontend and FastAPI/PyTorch backend for **research-only** classification of dermoscopic images into the seven HAM10000 categories:

- `akiec` — actinic keratoses / intraepithelial carcinoma
- `bcc` — basal cell carcinoma
- `bkl` — benign keratosis-like lesions
- `df` — dermatofibroma
- `mel` — melanoma
- `nv` — melanocytic nevi
- `vasc` — vascular lesions

> **Clinical safety notice:** This project is educational/research software, not a medical device. Its outputs are not diagnoses and must never be used as a basis for clinical decisions, treatment, or delay in seeking professional care.

## Dataset layout

Download HAM10000 yourself and place it outside the repository or in the ignored `data/` directory. The trainer discovers image files recursively, so the official `HAM10000_images_part_1` and `HAM10000_images_part_2` directories work unchanged.

```text
data/HAM10000/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
│   └── ISIC_*.jpg
└── HAM10000_images_part_2/
    └── ISIC_*.jpg
```

The metadata must have `image_id`, `lesion_id`, and `dx`. Splitting occurs by `lesion_id`, preventing multiple images of the same lesion from appearing in both training and validation sets.

## Backend setup and training

From `backend`:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py --data-dir ../data/HAM10000 --epochs 20 --batch-size 32
uvicorn app:app --reload --port 8000
```

Training writes the selected validation checkpoint to `backend/checkpoints/best.pt`. CUDA is used automatically when available. Tune epochs, batch size, and learning rate to your environment; evaluate carefully with clinically appropriate methods before any research use.

## Frontend setup

From `frontend`:

```bash
npm install
npm run dev
```

Vite proxies `/api` to `http://localhost:8000`. The UI reports that the model is unavailable until a checkpoint has been trained.

## Validation commands

```bash
# backend
python -m pytest -q

# frontend
npm run lint
npm run build
```

## API

- `GET /api/health` returns class codes and checkpoint availability.
- `POST /api/predict` accepts a `file` field containing a JPEG, PNG, or WebP image up to 10 MB. A trained checkpoint is required.

The service only returns model scores; callers must preserve the research-only disclaimer in any interface that displays them.
