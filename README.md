# Teammate Matcher

**Form balanced student teams from a quick survey — no spreadsheets, no guesswork.**

A web tool for UNC Charlotte instructors. Students fill out a short Google Form;
you upload the responses and get balanced teams optimized for compatible
schedules and work styles (or complementary skills). You stay in control — the
tool proposes teams, you make the final call.

> 🔗 **Live app:** https://teammate-matcher.streamlit.app
> 🧪 Built on the [`teammate-matcher`](../teammate-matcher) research project (DTSC 2302 capstone).

---

## What it does

- Takes anonymous student survey responses (identified only by **UNCC student ID**).
- Groups students into balanced teams using one of four models — **Hungarian
  assignment** (the recommended default) guarantees equal team sizes.
- Lets you pick the team size; the number of teams is derived automatically.
- Exports a `student_id → team` CSV you can join straight to your Canvas roster.

## How to use it (60-second version)

1. [**Make your own copy of the Google Form**](https://docs.google.com/forms/d/1n3zI__UIbtY5OZGqRnxfEafs6Zjb2Weq3V-NF7qthiA/copy) and share it with your class.
2. Download the responses as CSV (**Responses → ⋮ → Download .csv**).
3. Open the app, upload the CSV, pick a team size, and click **Generate teams**.
4. Download the results and match student IDs to names in your gradebook.

No account needed. Want to see it work first? Click **Try with example data**.

See [`docs/instructor_guide.md`](docs/instructor_guide.md) for the full walkthrough.

## Privacy

- Uploaded files are processed **in memory for your session only** — nothing is
  saved to a server or logged.
- Student IDs are used **purely as labels** — they never enter the matching
  algorithm.
- The survey collects no names or emails. Please don't upload files that do.
- A student ID is still a personal identifier (FERPA): don't post raw uploads
  publicly, and tell students on the form that their ID is collected.

---

## For developers

```bash
pip install -r requirements.txt
python scripts/generate_example_data.py   # regenerate demo data (optional)
streamlit run app.py
```

| Path | Purpose |
|---|---|
| `app.py` | Streamlit UI (upload → configure → results) |
| `src/preprocess.py` | Survey encoding pipeline + `process_dataframe()` (in-memory entry point) |
| `src/models.py` | Four matching models + `derive_k()` / `validate_n()` scaling helpers |
| `src/evaluate.py` | Six-metric team-quality evaluation |
| `scripts/generate_example_data.py` | Builds `assets/example_survey.csv` |

The ML logic is copied and generalized from the original capstone repo; that
project remains frozen as the research artifact.
