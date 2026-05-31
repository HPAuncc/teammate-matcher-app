"""
scripts/generate_example_data.py
--------------------------------
Generate a synthetic 24-response example survey CSV for the demo / "Try with
example data" button. Column headers and answer strings exactly match the
Google Forms template so the file flows through the real preprocessing pipeline.

No real students. Student IDs are sequential 800-numbers (UNCC format) and are
purely synthetic. Run from the repo root:

    python scripts/generate_example_data.py
"""

import csv
import random
from pathlib import Path

random.seed(42)

OUT = Path(__file__).resolve().parents[1] / "assets" / "example_survey.csv"

# ── Exact answer vocabularies (must match src/preprocess.py maps) ─────────────
YEARS = ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"]
HOURS = ["Less than 3 hours", "3–5 hours", "6–9 hours", "10+ hours"]
MEETING = ["In person only", "Remote only", "No preference"]
ROLES = [
    "I prefer to follow clear instructions",
    "I prefer to contribute as a specialist",
    "I'm comfortable in either role",
    "I prefer to lead and coordinate",
]
DEADLINES = [
    "I work best under pressure, closer to the deadline",
    "I work steadily throughout",
    "I finish well before the deadline",
]
COMM = ["Text/iMessage", "Email", "Discord/Slack", "Video call (Zoom, Teams)", "In person"]
CHECKIN = ["Only when necessary", "Once a week", "A few times a week", "Daily"]
COLLAB = [
    "Work independently and combine at the end",
    "A mix — divide tasks but check in regularly",
    "Collaborate closely throughout",
]
CONFLICT = [
    "Address it directly and immediately",
    "Talk it out privately first",
    "Let it resolve naturally",
    "Defer to whoever is leading",
]
GPA = ["Below 2.5", "2.5 – 3.0", "3.0 – 3.5", "3.5 – 4.0", "Prefer not to say"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TIMES = ["Morning", "Afternoon", "Evening", "Late night"]
CONTRIBS = [
    "Technical execution",
    "Creative ideas",
    "Organization and planning",
    "Research and writing",
    "Keeping team morale up",
    "Quality checking / editing",
]
PAIN = [
    "Coordinating schedules",
    "Unequal workload distribution",
    "Disagreements on direction",
    "Communication breakdowns",
    "Staying motivated",
]

HEADERS = [
    "Timestamp",
    "What is your UNCC student ID?",
    "What class is this team assignment for?",
    "What is your year in school?",
    "Which days are you generally available to meet with your team?",
    "What time(s) of day are you available?",
    "How many hours per week can you realistically dedicate to group project work?",
    "Do you prefer to meet in person or remotely?",
    "Python / programming",
    "Data analysis (pandas, spreadsheets, SQL)",
    "Statistics and math",
    "Data visualization (matplotlib, Tableau, etc.)",
    "Machine learning / modeling",
    "Technical writing and documentation",
    "Research and literature review",
    "Presentations and public speaking",
    "What role do you naturally gravitate toward in a group?",
    "How do you typically approach deadlines?",
    "How do you prefer to communicate with your team? (pick primary)",
    "How often do you prefer to check in with teammates?",
    "When working on a group project, I prefer to...",
    "I tend to focus on...",
    "My approach to conflict in a team is...",
    "What is your GPA range? (Optional)",
    "What do you most contribute to a team? (Select up to 2)",
    "What is your biggest challenge in group projects?",
]

# Three loose "archetypes" give the clustering something real to find:
#  A: weekday/morning, strong coding + stats, early/steady workers
#  B: evening/weekend, strong writing + presenting + research, collaborative
#  C: flexible, balanced skills, mixed work style
ARCHETYPES = ["A"] * 8 + ["B"] * 8 + ["C"] * 8
random.shuffle(ARCHETYPES)


def skills_for(arch):
    """Return the 8 skill ratings (1-5) for an archetype, with noise."""
    def j(center):
        return max(1, min(5, center + random.choice([-1, 0, 0, 1])))
    if arch == "A":  # technical
        return [j(5), j(5), j(4), j(3), j(4), j(2), j(2), j(2)]
    if arch == "B":  # communication / writing
        return [j(2), j(3), j(2), j(3), j(2), j(5), j(5), j(5)]
    return [j(3), j(3), j(3), j(3), j(3), j(3), j(3), j(3)]  # balanced


def days_for(arch):
    if arch == "A":
        pool = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    elif arch == "B":
        pool = ["Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    else:
        pool = DAYS
    k = random.randint(2, 4)
    return ", ".join(sorted(random.sample(pool, k), key=DAYS.index))


def times_for(arch):
    if arch == "A":
        pool = ["Morning", "Afternoon"]
    elif arch == "B":
        pool = ["Evening", "Late night"]
    else:
        pool = TIMES
    k = random.randint(1, 2)
    return ", ".join(sorted(random.sample(pool, k), key=TIMES.index))


rows = []
for i, arch in enumerate(ARCHETYPES):
    sid = 800_000_001 + i
    skills = skills_for(arch)
    n_contrib = random.randint(1, 2)
    contribs = ", ".join(random.sample(CONTRIBS, n_contrib))
    rows.append([
        f"2026/05/15 {9 + i % 12}:{i * 2 % 60:02d}:00 AM",  # plausible timestamp
        sid,
        "DTSC 2302",
        random.choice(YEARS),
        days_for(arch),
        times_for(arch),
        random.choice(HOURS[1:]) if arch != "C" else random.choice(HOURS),
        random.choice(MEETING),
        skills[0], skills[1], skills[2], skills[3],
        skills[4], skills[5], skills[6], skills[7],
        random.choice(ROLES),
        "I finish well before the deadline" if arch == "A" else random.choice(DEADLINES),
        random.choice(COMM),
        random.choice(CHECKIN),
        "Collaborate closely throughout" if arch == "B" else random.choice(COLLAB),
        random.randint(1, 5),  # detail_orientation ("I tend to focus on...")
        random.choice(CONFLICT),
        random.choice(GPA),
        contribs,
        random.choice(PAIN),
    ])

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(HEADERS)
    w.writerows(rows)

print(f"Wrote {len(rows)} rows -> {OUT}")
