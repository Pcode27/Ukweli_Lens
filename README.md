# Ukweli_Lens

Truth Verifier (Fact-Checking Engine)s.

## Overview

This repository contains the Ukweli_Lens project: a multilingual AI fact-checking engine built for Kenya’s unique linguistic and cultural landscape. It detects and verifies misinformation in English, Kiswahili, and Sheng, with a special focus on Technology-Facilitated Gender-Based Violence (TFGBV) and high-risk political/ethnic incitement.

## Quick start

1. Create and activate a Python virtual environment.

PowerShell (recommended):

```powershell
cd D:\SAFEAI\Project\ukweli_lens
python -m venv venv
. .\venv\Scripts\Activate.ps1
```

Command Prompt (cmd.exe):

```cmd
cd /d D:\SAFEAI\Project\ukweli_lens
venv\Scripts\activate.bat
```

macOS / Linux / WSL:

```bash
cd /mnt/d/SAFEAI/Project/ukweli_lens
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies

```powershell
pip install -r Project\ukweli_lens\requirements.txt
```

3. Run the development server

```powershell
cd Project\ukweli_lens
python manage.py migrate
python manage.py runserver
```

4. API

Example verify endpoint (POST):

```powershell
Invoke-WebRequest -Uri http://127.0.0.1:8000/api/verify/ -Method POST -ContentType "application/json" -Body '{"claim": "Your claim text here"}'
```

## Files to keep locally

Sensitive/large local files are intentionally untracked. Examples:

- `PK Data Release Agreement.pdf`
- `Problem Statement.docx`
- `SAFE AI Teams.pdf`
- `Team_FactCheckingEngine_Track_[1].docx`

These files remain on local disk but are ignored by git.

## Contributing

- Make small, focused commits.
- Create feature branches: `git checkout -b feature/your-feature`.
- Open pull requests against `master`.

## License

Add a license as appropriate.

---

If you'd like, I can add a `requirements.txt` or minimal contributor guidelines next.
