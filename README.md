# Hackathon Student Support API

FastAPI service and CLI for analyzing student-written text for academic stress, mental-health risk, and dominant emotions. The project ships with fine-tuned transformer models, a secure student submission flow (with roster validation and HIPAA consent), a therapist review dashboard, and encrypted high-risk alert handling.

## What’s here

- `AI Model` - There was not 20 GB worth of space on this repository for the model
- `HackathonProject1_updated.py` – main FastAPI app, CLI entry point, and stress-model fine-tuning utility.
- `models/` – bundled checkpoints (roberta-based stress model and a multi-label mental-health classifier).
- `data/student_roster.csv` – sample roster used for ID validation, class names, and extra-credit tracking.
- `data/therapists.csv` – demo therapist/police logins for the review board dashboard.
- `data/mental_train.csv`, `data/mental_val.csv` – example splits for mental-health training.
- `load_test_mental_val.py` – concurrency/load tester for `POST /analyze`.
- `prepare_mental_labels.py`, `train_mental_health.py` – helper scripts to build and fine-tune the mental-health model.
- `alerts/` – encrypted alert payloads (.enc) and keys (.key) generated for high-risk submissions.

## Quickstart

1. Python 3.10+ is recommended. Create a virtualenv and install deps:
   ```pwsh
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install fastapi uvicorn[standard] transformers datasets torch pandas numpy scikit-learn requests cryptography
   ```
   (Install the appropriate CPU/GPU build of PyTorch for your machine.)
2. Models are already in `models/`; no download needed. If you prefer to pull from Hugging Face, set `HACKATHON_STRESS_BASE`, `HACKATHON_MENTAL_MODEL`, and `HACKATHON_EMOTION_MODEL` env vars.
3. Run the API:
   ```pwsh
   python HackathonProject1_updated.py --serve --host 0.0.0.0 --port 8000
   ```
4. Open `http://localhost:8000/docs` for the custom “Try it out” UI. Validate a student ID, paste text (min words enforced), consent to HIPAA, then execute.

## API surface

- `GET /` – health tip with usage pointers.
- `POST /validate-student` – body `{ "student_id": "…" }`; validates against `data/student_roster.csv`, returns roster classes and extra-credit state.
- `POST /analyze` – body `{ "student_id": "...", "text": "...", "consent": true }`; returns:
  - `academic_stress` (0–5 scale),
  - `mental_health` (multi-label %: depression, anxiety, loneliness, stress, suicidal, etc.),
  - `emotions` (emotion %),
  - `mental_health_flags` and optional `alert_metadata` when thresholds/keywords indicate risk.
- `POST /extra-credit` – `{ "student_id", "class_key", "points": 1 }`; grants a single +1 extra-credit entry per student.
- `GET /submit` – lightweight web form for students (writes analysis to console).
- `GET /therapist` – auto-refreshing review board; requires a therapist login from `data/therapists.csv`.
- `GET /therapist-login`, `POST /therapist-login`, `POST /therapist-logout` – dashboard auth flow.

Example request:
```pwsh
curl -X POST http://localhost:8000/analyze `
  -H "Content-Type: application/json" `
  -H "X-API-Key: $env:HACKATHON_API_KEY" `
  -d '{"student_id":"1","text":"I am overwhelmed by finals and feel hopeless.","consent":true}'
```

## CLI mode

Run without `--serve` to analyze text directly in terminal:
```pwsh
python HackathonProject1_updated.py
# type a message and press Enter; type 'quit' to exit
```

## Training options

- **Stress model (roberta large)**: fine-tune before serving with `--train`:
  ```pwsh
  python HackathonProject1_updated.py `
    --train `
    --train-file data/mental_train.csv `
    --eval-file data/mental_val.csv `
    --text-column text `
    --label-column label `
    --output-dir models/stress-model
  ```
  Optional helpers: `--augment-file` (extra CSV/JSON), `--dataset-root` (StudentLife pairing), `--train-csv` to force pandas loader.
- **Mental-health multi-label model**: heuristic label builder + trainer:
  ```pwsh
  python prepare_mental_labels.py --input-file "dreaddit_StressAnalysis - Sheet1.csv"
  python train_mental_health.py `
    --train-file data/mental_train.csv `
    --eval-file data/mental_val.csv `
    --label-columns stress,depression,suicidal `
    --base-model distilroberta-base `
    --output-dir models/mental-model
  ```

## Load testing

Blast the service with the validation split:
```pwsh
python load_test_mental_val.py --url http://localhost:8000/analyze --concurrency 25 --api-key $env:HACKATHON_API_KEY
```
The script cycles roster IDs from `data/student_roster.csv` (or a fixed `--student-id`) and reports throughput/latency.

## Data, auth, and alerts

- **Roster**: IDs, names, classes, HIPAA consent, and extra-credit tallies live in `data/student_roster.csv`. The app writes back to this file when consent or extra credit changes.
- **Therapist dashboard**: credentials sourced from `data/therapists.csv` (`username,password,first_responder`). Default demo users: `therapist/password` and `police/password`.
- **High-risk encryption**: when suicidal indicators trip, the raw text is encrypted with Fernet into `alerts/*.enc` with a paired `*.key`. Share keys only through secure channels. Disable/enable with `HACKATHON_ENCRYPT_ON_ALERT` (defaults on; requires `cryptography`).

## Environment variables

- `HACKATHON_API_KEY` – require `X-API-Key` on student and extra-credit routes.
- `HACKATHON_ALLOWED_ORIGINS` – CORS allowlist (comma-separated, default `*`).
- `HACKATHON_MIN_RESPONSE_WORDS` – minimum word count (default 50).
- `HACKATHON_STRESS_MODEL`, `HACKATHON_STRESS_BASE`, `HACKATHON_STRESS_TOKENIZER` – override stress model paths/names.
- `HACKATHON_MENTAL_MODEL`, `HACKATHON_EMOTION_MODEL` – override mental/emotion checkpoints.
- `HACKATHON_MENTAL_THRESHOLD` – multi-label activation threshold % (default 50), `HACKATHON_MENTAL_MULTI_LABEL` to force on/off.
- `HACKATHON_SUICIDAL_AUTO_THRESHOLD` – depression fallback threshold when suicide keywords present (default 70).
- `HACKATHON_HISTORY_LIMIT` – length of in-memory dashboard history (default 50).
- `HACKATHON_STUDENT_ROSTER`, `HACKATHON_THERAPIST_CSV` – alternate CSV paths.
- `HACKATHON_ALERT_DIR` – where to write `.enc/.key` files (default `alerts/`).
- `HACKATHON_ENCRYPT_ON_ALERT` – turn alert encryption on/off (default on).

## Tips

- Use the `/docs` UI for end-to-end flow: validate ID → enter text → optional API key → submit (extra credit is applied automatically when configured).
- Keep `alerts/` and `data/` writable where the service runs; extra-credit and consent updates persist to CSV.
- If you swap checkpoints, ensure label order matches expectations (see `label_mapping.json` in `models/stress-model` for the stress labels).
