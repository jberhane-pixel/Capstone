#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, csv, hashlib
from datetime import date

# ======================
# Settings
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder that contains resumes (pdf / docx / txt)
INPUT_DIR = os.path.join(BASE_DIR, "resumes")

OUTPUT_CSV  = os.path.join(BASE_DIR, "scanned_resumes.csv")
OUTPUT_JSON = os.path.join(BASE_DIR, "scanned_resumes.json")

# ======================
# Helpers
# ======================
def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# ======================
# File readers
# ======================
def read_pdf(path: str) -> str:
    import pdfplumber
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                text.append(t)
    return "\n".join(text)

def read_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_any(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".docx":
        return read_docx(path)
    if ext == ".txt":
        return read_txt(path)
    return ""

# ======================
# Extraction helpers
# ======================
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}"
)
LINK_RE = re.compile(r"https?://\S+|www\.\S+", re.I)

SECTION_HEADINGS = {
    "summary", "profile", "objective",
    "education",
    "experience", "work experience", "employment",
    "projects",
    "skills", "technical skills",
    "certifications",
    "awards", "honors",
    "publications",
    "volunteer", "volunteering",
    "activities", "leadership"
}

def guess_name(text: str) -> str:
    lines = [norm_space(l) for l in text.splitlines() if l.strip()]
    for l in lines[:12]:
        if EMAIL_RE.search(l) or PHONE_RE.search(l):
            continue
        if l.lower() in {"resume", "curriculum vitae", "cv"}:
            continue
        if l.isupper() and len(l) >= 8:
            continue
        if 1 < len(l.split()) <= 4 and sum(c.isalpha() for c in l) >= len(l) * 0.6:
            return l
    return ""

def extract_contacts(text: str) -> dict:
    emails = list(dict.fromkeys(EMAIL_RE.findall(text)))
    phones = list(dict.fromkeys(PHONE_RE.findall(text)))
    links  = list(dict.fromkeys(LINK_RE.findall(text)))

    cleaned_links = []
    for l in links:
        l = l.rstrip(".,;")
        if l.lower().startswith("www."):
            l = "https://" + l
        cleaned_links.append(l)

    return {
        "emails": emails,
        "phones": phones,
        "links": cleaned_links
    }

def split_sections(text: str) -> dict:
    lines = [norm_space(l) for l in text.splitlines()]
    sections = {}
    current = "header"
    buf = []

    def flush():
        nonlocal buf
        content = "\n".join(buf).strip()
        if content:
            sections[current] = (sections.get(current, "") + "\n" + content).strip()
        buf = []

    for l in lines:
        t = l.lower().strip(": ")
        if t in SECTION_HEADINGS:
            flush()
            current = t
        else:
            buf.append(l)

    flush()
    return sections

def extract_skills(sections: dict) -> list[str]:
    skills_text = ""
    for k, v in sections.items():
        if "skill" in k:
            skills_text += "\n" + v

    if not skills_text.strip():
        return []

    parts = re.split(r"[•\u2022,\n;/|]+", skills_text)
    cleaned = []
    for p in parts:
        it = norm_space(p)
        if 1 <= len(it) <= 40:
            cleaned.append(it)

    return list(dict.fromkeys(cleaned))

# ======================
# Main scan
# ======================
def iter_files(folder: str):
    for root, _, files in os.walk(folder):
        for fn in files:
            if fn.lower().endswith((".pdf", ".docx", ".txt")):
                yield os.path.join(root, fn)

def scan_file(path: str) -> dict:
    raw = read_any(path).replace("\x00", "").strip()
    sections = split_sections(raw)
    contacts = extract_contacts(raw)
    skills = extract_skills(sections)

    return {
        "resume_id": "RES-" + file_sha1(path)[:12],
        "file_name": os.path.basename(path),
        "file_type": os.path.splitext(path)[1].lower(),
        "name_guess": guess_name(raw),
        "emails": "; ".join(contacts["emails"]),
        "phones": "; ".join(contacts["phones"]),
        "links": "; ".join(contacts["links"]),
        "skills_guess": "; ".join(skills),
        "sections_json": json.dumps(sections, ensure_ascii=False),
        "resume_text": norm_space(raw),
        "collected_date": str(date.today()),
    }

def main():
    if not os.path.isdir(INPUT_DIR):
        print(f"Missing folder: {INPUT_DIR}")
        print("Create a folder named 'resumes' next to this script.")
        return

    rows = []
    for path in iter_files(INPUT_DIR):
        print("\nScanning:", path)
        try:
            r = scan_file(path)
            print("  name:", r["name_guess"] or "(none)")
            print("  text_len:", len(r["resume_text"]))
            rows.append(r)
        except Exception as e:
            print("  ERROR:", e)

    if not rows:
        print("\nNo files scanned.")
        return

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    out = []
    for r in rows:
        rr = dict(r)
        rr["sections"] = json.loads(rr.pop("sections_json", "{}"))
        out.append(rr)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(rows)} files")
    print("→", OUTPUT_CSV)
    print("→", OUTPUT_JSON)

if __name__ == "__main__":
    main()
