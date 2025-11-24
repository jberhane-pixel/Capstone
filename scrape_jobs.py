#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import json, re, csv, os
from datetime import date
import os

# ==========  
# Settings  
# ==========

# Get the folder where THIS script is located  
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use files from this same folder  
PATH_TO_URLS = os.path.join(BASE_DIR, "job_urls.txt")
OUTPUT_CSV   = os.path.join(BASE_DIR, "scraped_jobs.csv")
OUTPUT_JSON  = os.path.join(BASE_DIR, "scraped_jobs.json")
# ==============================================
# Helpers: text cleanup + section parsing
# ==============================================

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

WHITELIST_HEADINGS = [
    "responsibilities", "what you will do", "what you'll do", "what you will work on",
    "role", "about the role", "job description", "description",
    "requirements", "qualifications", "minimum qualifications", "basic qualifications",
    "preferred", "preferred qualifications", "nice to have", "nice-to-have"
]
BLACKLIST_HEADINGS = [
    "about", "about us", "company", "our mission", "who we are",
    "benefits", "perks", "why join", "eeo", "equal opportunity",
    "privacy", "accommodation", "legal", "notice", "disclaimer"
]

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def is_heading(text: str) -> bool:
    t = text.lower().strip(": ").strip()
    if len(t) > 80:   # long lines are not headings
        return False
    return any(k in t for k in WHITELIST_HEADINGS + BLACKLIST_HEADINGS)

def split_sections_from_html(html: str):
    """
    Try to split a job page into {heading -> body} sections using headings and bold labels.
    Falls back to a single 'generic' section.
    """
    soup = BeautifulSoup(html, "html.parser")
    sections = []
    candidates = []

    # Headings
    for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        text = norm_space(h.get_text(" ", strip=True))
        if is_heading(text):
            candidates.append((h, text.lower()))

    # Bold labels that look like headings
    for b in soup.find_all(["strong", "b"]):
        text = norm_space(b.get_text(" ", strip=True))
        if is_heading(text):
            candidates.append((b, text.lower()))

    seen = set()
    ordered = []
    for tag, label in candidates:
        if id(tag) not in seen:
            seen.add(id(tag))
            ordered.append((tag, label))

    if not ordered:
        main = soup.find("main") or soup.find("article") or soup
        generic_text = norm_space(main.get_text(" ", strip=True))
        return {"generic": generic_text}

    # slice content between headings
    for idx, (tag, label) in enumerate(ordered):
        body_parts = []
        cur = tag.next_sibling
        stop_at = ordered[idx + 1][0] if idx + 1 < len(ordered) else None
        while cur and cur is not stop_at:
            if hasattr(cur, "get_text"):
                body_parts.append(cur.get_text(" ", strip=True))
            cur = cur.next_sibling
        body = norm_space(" ".join(body_parts))
        if body:
            sections.append((label, body))

    merged = {}
    for label, body in sections:
        merged[label] = (merged.get(label, "") + " " + body).strip()
    return merged

def keep_only_job_content(sections: dict) -> str:
    """
    Keep whitelisted sections and drop blacklisted ones; if nothing whitelisted,
    return the largest remaining chunk that doesn't look like EEO/benefits/privacy.
    """
    kept = []

    for label, text in sections.items():
        if any(k in label for k in WHITELIST_HEADINGS):
            kept.append(text)

    if not kept:
        candidates = []
        for label, text in sections.items():
            if not any(k in label for k in BLACKLIST_HEADINGS):
                candidates.append((len(text), text))
        if candidates:
            candidates.sort(reverse=True)  # largest first
            kept.append(candidates[0][1])

    blob = " ".join(kept)
    blob = re.sub(r"equal opportunity employer.*?(?=\. )", "", blob, flags=re.I)
    blob = re.sub(r"accommodations? .*?(?=\. )", "", blob, flags=re.I)
    blob = re.sub(r"privacy policy.*?(?=\. )", "", blob, flags=re.I)
    return norm_space(blob)

# ==============================================
# Extractors
# ==============================================

def extract_jsonld_jobposting(soup: BeautifulSoup):
    """Generic JSON-LD JobPosting extractor (title/company/location/description)."""
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
            blobs = data if isinstance(data, list) else [data]
            for b in blobs:
                tp = b.get("@type")
                if tp == "JobPosting" or (isinstance(tp, list) and "JobPosting" in tp):
                    title = (b.get("title") or "").strip()
                    org = b.get("hiringOrganization")
                    company = org.get("name").strip() if isinstance(org, dict) else ""
                    # location
                    loc = ""
                    jl = b.get("jobLocation")
                    if isinstance(jl, list) and jl:
                        jl = jl[0]
                    if isinstance(jl, dict):
                        addr = jl.get("address") or {}
                        loc = " ".join(filter(None, [
                            addr.get("addressLocality"),
                            addr.get("addressRegion"),
                            addr.get("addressCountry"),
                        ])).strip()
                    desc_html = b.get("description") or ""
                    desc = keep_only_job_content(split_sections_from_html(desc_html)) if desc_html else ""
                    return title, company, loc, desc
        except Exception:
            continue
    return "", "", "", ""

def extract_workday(html: str):
    """Workday pages: JSON-LD first, then the big internal JSON blob."""
    soup = BeautifulSoup(html, "html.parser")
    title, company, loc, desc = extract_jsonld_jobposting(soup)
    if title or desc:
        return title, company, loc, desc

    # Fallback: search big Workday JSON
    m = re.search(r'(\{.*"jobPostingInfo".*\})', html, flags=re.DOTALL)
    if m:
        blob = m.group(1)
        end = blob.rfind("}")
        try:
            data = json.loads(blob[:end+1])
            jpi = data.get("jobPostingInfo", {})
            title = (jpi.get("title") or "").strip()
            company = (jpi.get("hiringCompany", {}).get("name") or "").strip()
            loc = jpi.get("location", "")
            desc_html = jpi.get("jobDescription") or ""
            desc = keep_only_job_content(split_sections_from_html(desc_html))
            return title, company, loc, desc
        except Exception:
            pass

    # Final fallback: og:title + filtered page text
    og = soup.find("meta", property="og:title")
    title = og.get("content").strip() if og and og.has_attr("content") else ""
    filtered = keep_only_job_content(split_sections_from_html(html))
    return title, "", "", filtered

def extract_amd(html: str):
    """
    AMD careers pages: try JSON-LD; fallback to AMD-specific content blocks.
    AMD pages often include clean content inside main/article roles.
    """
    soup = BeautifulSoup(html, "html.parser")
    # Try JSON-LD JobPosting first
    title, company, loc, desc = extract_jsonld_jobposting(soup)
    if not company:
        # If it’s AMD, fill company
        company = "Advanced Micro Devices, Inc" if "careers.amd.com" in (soup.base and soup.base.get("href","") or "") or True else company

    if title or desc:
        return title, company, loc, desc

    # Heuristics: main/article and common AMD blocks
    main = soup.find("main") or soup.find("article") or soup
    # Some AMD pages show the description in obvious containers
    candidates = []
    for sel in [
        "div.job-description", "section.job-description", "div#job-description",
        "div.description", "section.description", "div[class*=description]",
        "div[class*=job-content]", "div[class*=jobDetails]"
    ]:
        for el in main.select(sel):
            text = norm_space(el.get_text(" ", strip=True))
            if len(text) > 200:
                candidates.append((len(text), text))
    if not candidates:
        # fallback to largest text in main
        text = norm_space(main.get_text(" ", strip=True))
        candidates.append((len(text), text))

    candidates.sort(reverse=True)
    raw = candidates[0][1]
    filtered = keep_only_job_content(split_sections_from_html(raw))
    # Title fallback
    meta_t = soup.find("meta", property="og:title")
    title2 = (meta_t.get("content").strip() if meta_t and meta_t.has_attr("content")
              else soup.find("title").get_text(strip=True) if soup.find("title") else "")
    return title2 or title, company, loc, filtered

def extract_generic(html: str):
    """Generic extractor for anything else."""
    soup = BeautifulSoup(html, "html.parser")
    title, company, loc, desc = extract_jsonld_jobposting(soup)
    if title or desc:
        return title, company, loc, desc

    # Title guess
    meta_t = soup.find("meta", property="og:title")
    title = (meta_t.get("content").strip() if meta_t and meta_t.has_attr("content")
             else soup.find("title").get_text(strip=True) if soup.find("title") else "No Title Found")

    # Company guess
    meta_c = soup.find("meta", {"name": "company"}) or soup.find("meta", {"property": "og:site_name"})
    company = (meta_c.get("content").strip() if meta_c and meta_c.has_attr("content") else "")

    # Description: take meaningful sections only
    filtered = keep_only_job_content(split_sections_from_html(html))

    # Location heuristic: look for "Location:"
    full = soup.get_text(" ", strip=True)
    mloc = re.search(r"\bLocation\s*:\s*([^\n|]+)", full, re.I)
    loc = norm_space(mloc.group(1)) if mloc else ""

    return title, company, loc, filtered

# ==============================================
# Load URLs
# ==============================================
with open(PATH_TO_URLS, "r", encoding="utf-8") as f:
    URLS = [ln.strip() for ln in f if ln.strip()]

# ==============================================
# Run
# ==============================================
rows = []
for url in URLS:
    print("\nFetching:", url)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        html = resp.text
    except Exception as e:
        print("  ERROR fetching:", e)
        continue

    title = company = loc = desc = ""

    if "careers.amd.com" in url:
        title, company, loc, desc = extract_amd(html)
    elif "myworkdayjobs" in url:
        title, company, loc, desc = extract_workday(html)
    else:
        title, company, loc, desc = extract_generic(html)

    # Extract entire page text as the full job description
    soup_full = BeautifulSoup(html, "html.parser")
    full_desc = norm_space(soup_full.get_text(" ", strip=True))

    # Choose the best available description: extractor result (desc) if present,
    # otherwise fall back to the full page text we already collected.
    chosen_desc = desc.strip() or full_desc

    print("Title:     ", title or "(none)")
    print("Company:   ", company or "(unknown)")
    print("Location:  ", loc or "(unknown)")
    print("Desc len:  ", len(chosen_desc))
    print("Desc snip: ", (chosen_desc[:1000] + "...") if chosen_desc else "(none)")

    rows.append({
        "job_id": f"URL-{len(rows)+1:04d}",
        "title": title,
        "company": company,
        "location": loc,
        "employment_type": "",      # can guess later
        "seniority": "",            # can guess later
        "domain": "",               # can guess later
        "education_min": "",
        "years_min": "",
        "salary_range": "",
        "must_have_skills": "",     # fill in your taxonomy pass later
        "nice_to_have_skills": "",
        "certifications": "",
        "description_text": chosen_desc,   # full job posting text when available
        "source_url": url,
        "posted_date": "",
        "collected_date": str(date.today())
    })

# ==============================================
# Save CSV
# ==============================================
if rows:
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nSaved {len(rows)} rows → {OUTPUT_CSV}")



# ==============================================
# Save JSON
# ==============================================
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(rows)} rows → {OUTPUT_JSON}")

else:
    print("\nNo rows scraped. Check URLs or site blocking.")
    