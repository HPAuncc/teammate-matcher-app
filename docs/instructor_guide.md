# Instructor's Guide — Teamora

A step-by-step guide to forming student teams with Teamora. No
technical background needed.

---

## Before you start

You'll need your own copy of the Google Form to collect responses. Use the
official template so the columns line up with the tool:

- **Get your own copy:** https://docs.google.com/forms/d/1n3zI__UIbtY5OZGqRnxfEafs6Zjb2Weq3V-NF7qthiA/copy
  - Sign in with your **`@charlotte.edu`** account first — the template is
    shared with UNC Charlotte only, so the link won't copy from a personal
    account.
  - Clicking it then prompts you to **save a private copy to your own Google
    Drive** — your own form, your own responses. It never touches anyone else's
    copy, and you don't need edit access to the original.
- Your copy will be named **"Copy of Team Formation Survey — Instructor
  Template."** Rename it to your course/section (e.g., *DTSC 2302 Spring 26
  Team Survey*) so your responses file is clearly labeled. This is just the
  form's title — **don't** rename the individual questions (see below).
- In your copy, keep these settings:
  - **Do not** collect email addresses — this keeps the form pseudonymous
    (student ID only), which is the privacy basis of the tool.
  - **Limit to 1 response** is recommended (stops duplicate submissions). Note
    this *does* require students to sign in to a Google account — their
    `@charlotte.edu` account works fine, and their email is still not collected.
  - **Allow response editing** is recommended, so a student can fix a typo
    instead of submitting twice.
  - Keep the **UNCC student ID** question required — it's how teams map back to
    your roster.
  - Don't rename the fixed questions — the tool matches them by their exact
    wording. The **one part you *should* customize is the skills grid** (see
    "Tailor the skills to your course" below).

Aim for **at least 9 responses** (ideally 20+) for the matching to be meaningful.

---

## Tailor the skills to your course

The survey includes **one skill question — a grid** where students rate
themselves 1–5 on each skill. This is the part you make your own:

- **Edit the grid's rows** to the skills that matter for *your* course — e.g. a
  writing seminar: *Close reading, Argumentation, Editing, Research*; an
  engineering course: *CAD, Circuits, Prototyping, Lab safety*.
- **Keep the 1–5 column scale** — the quality scores depend on it.
- Add or remove rows freely (**5–10** works best; up to ~12). Teamora reads
  whatever skills you put there and labels the results with your exact wording —
  no code changes needed.
- **Don't rename the other questions**, and don't change the grid's *columns*
  (keep them 1–5).
- **No skills needed?** Delete the grid entirely. Teamora detects there are no
  skills and matches purely on **schedule and work style** — it hides the
  skills-based model and the skill charts automatically.

---

## Step 1 — Collect responses

Share **your copy's** link with your class (Canvas announcement, email, etc.).
Give students a deadline so you have all responses before forming teams.

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
   - **K-Means / Agglomerative** — also compatibility-based. (All models keep
     team sizes even by default.)
   - **Gaussian Mixture** — groups by *complementary skills* so no team is
     uniformly weak in one area. (Only shown when your survey includes skills.)
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
You're meant to customize the **skills grid** — its rows are yours to set. The
other questions are matched by their exact wording; if one is renamed or
removed, Teamora tells you exactly which one so you can restore it.

**Can I see what it does before using real data?**
Yes — click **Try with example data** on the upload screen to run it on a
synthetic class of 24 students.

**My class is very large / very small.**
Below ~9 students the tool will warn you that clustering isn't reliable. Very
large classes (150+) work but may take a moment.
