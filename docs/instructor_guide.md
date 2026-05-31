# Instructor's Guide — Teammate Matcher

A step-by-step guide to forming student teams with Teammate Matcher. No
technical background needed.

---

## Before you start

You'll need a Google Form to collect responses. Use the official template so the
columns line up with the tool:

- **Template form:** https://forms.gle/i3ZyERWofpMMfuH78
- Make a copy (**⋮ → Make a copy** or **Send a copy to students**), and keep
  these settings:
  - **Do not** collect email addresses.
  - **Do not** require Google sign-in.
  - Keep the **UNCC student ID** question required — it's how teams map back to
    your roster.
  - Don't rename the questions. The tool matches them by their exact wording.

Aim for **at least 9 responses** (ideally 20+) for the matching to be meaningful.

---

## Step 1 — Collect responses

Share the form link with your class (Canvas announcement, email, etc.). Give
students a deadline so you have all responses before forming teams.

## Step 2 — Download the CSV

In your form:

1. Open the **Responses** tab.
2. Click the **⋮** (three dots) menu.
3. Choose **Download responses (.csv)**.

You'll get a file like `Team Survey (Responses).csv`.

## Step 3 — Generate teams

1. Open the app (link in the README / your bookmarked URL).
   - First visit after a quiet period may take ~30 seconds to wake up — that's
     normal.
2. **Upload** the CSV you just downloaded.
3. Choose a **preferred team size** (3–6). The tool tells you how many teams
   that produces.
4. Choose a **model**:
   - **Hungarian (recommended)** — equal-sized teams matched on schedule and
     work-style compatibility. Start here.
   - **K-Means / Agglomerative** — also compatibility-based, but team sizes can
     vary.
   - **Gaussian Mixture** — groups by *complementary skills* so no team is
     uniformly weak in one area (sizes vary).
5. Click **Generate teams**.

## Step 4 — Use the results

- Review the teams on screen.
- Click **Download results (.zip)** to get:
  - `team_assignments.csv` — each `student_id` with its team number.
  - `evaluation_metrics.csv` — quality scores for the grouping.
- In your gradebook/Canvas roster, match each **student ID** to a name to build
  your final groups.

You always have the final say — adjust any team by hand if you know context the
survey didn't capture.

---

## Frequently asked

**Is student data stored anywhere?**
No. Your uploaded file is processed in memory for that session only and is
discarded when you close the tab. Nothing is saved or logged. Student IDs are
used only to label the output — they never affect how teams are formed.

**What if I changed the survey questions?**
The tool matches questions by their exact wording. If you rename or remove
questions, the matching columns won't be found and you'll see an error. Keep the
template wording, or reach out to have the tool updated.

**Can I see what it does before using real data?**
Yes — click **Try with example data** on the upload screen to run it on a
synthetic class of 24 students.

**My class is very large / very small.**
Below ~9 students the tool will warn you that clustering isn't reliable. Very
large classes (150+) work but may take a moment.
