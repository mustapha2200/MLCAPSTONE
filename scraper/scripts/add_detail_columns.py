#!/usr/bin/env python3
"""
Script to add Overview & Details columns to real_estate_listings.csv
This script will enrich the CSV with data from detail pages if not already present.
"""

import pandas as pd
import requests
import json
import time
import sys
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional

# Optional: limit number of listings for testing (set via command line: python script.py 20)
MAX_LISTINGS = None
if len(sys.argv) > 1:
    try:
        MAX_LISTINGS = int(sys.argv[1])
        print(f"TEST MODE: Limiting to {MAX_LISTINGS} listings")
    except:
        pass

# Load existing CSV
print("Loading existing CSV...")
df = pd.read_csv("real_estate_listings.csv")
print(f"Loaded {len(df)} listings")

# Check if detail columns already exist
detail_cols = ["lot_size", "street_size", "closest_landmark", "land_title",
               "detail_property_type", "halls", "balconies", "bathrooms", "rooms"]
existing_cols = [c for c in detail_cols if c in df.columns]
missing_cols = [c for c in detail_cols if c not in df.columns]

if not missing_cols:
    print("All detail columns already exist in CSV!")
    exit(0)

print(f"\nExisting detail columns: {existing_cols}")
print(f"Missing detail columns: {missing_cols}")

# Initialize detail columns with None
for col in missing_cols:
    df[col] = None

# Setup session
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,fr;q=0.8,ar;q=0.7",
}

session = requests.Session()
session.headers.update(DEFAULT_HEADERS)

def parse_voursa_detail_html(html: str) -> Dict[str, Any]:
    """Extract overview and details from a Voursa listing detail page."""
    out: Dict[str, Any] = {
        "lot_size": None,
        "street_size": None,
        "closest_landmark": None,
        "land_title": None,
        "detail_property_type": None,
        "halls": None,
        "balconies": None,
        "bathrooms": None,
        "rooms": None,
    }

    idx = html.find('"overview"')
    if idx < 0:
        idx = html.find('\\"overview\\"')
    if idx < 0:
        return out

    start = html.rfind("{", 0, idx + 1)
    if start < 0:
        return out

    depth = 0
    end = start
    for i in range(start, len(html)):
        c = html[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    try:
        raw = html[start : end + 1]
        raw = raw.replace('\\"', '"')
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return out

    for item in data.get("overview") or []:
        key = (item.get("key") or "").strip()
        val = (item.get("value") or "").strip()
        if key == "Lot Size":
            out["lot_size"] = val if val and val.upper() != "N/A" else None
        elif key == "Street Size":
            out["street_size"] = val if val and val.upper() != "N/A" else None
        elif key == "Closest Landmark":
            out["closest_landmark"] = val or None

    for item in data.get("details") or []:
        key = (item.get("key") or "").strip()
        val = (item.get("value") or "").strip()
        if key == "Land Title":
            out["land_title"] = val or None
        elif key == "Property Type":
            out["detail_property_type"] = val or None
        elif key == "Halls":
            out["halls"] = int(val) if val and val.isdigit() else None
        elif key == "Balconies":
            out["balconies"] = int(val) if val and val.isdigit() else None
        elif key == "Bathrooms":
            out["bathrooms"] = int(val) if val and val.isdigit() else None
        elif key == "Rooms":
            out["rooms"] = int(val) if val and val.isdigit() else None

    return out

# Enrich only Voursa listings that don't have detail data yet
voursa_urls = df[df["source"] == "voursa"]["url"].drop_duplicates().tolist()
if MAX_LISTINGS is not None:
    voursa_urls = voursa_urls[:MAX_LISTINGS]
    print(f"\nTEST MODE: Enriching {len(voursa_urls)} Voursa listings (limited from {len(df[df['source'] == 'voursa']['url'].drop_duplicates())} total)...")
else:
    print(f"\nEnriching {len(voursa_urls)} Voursa listings...")
    print("This will take approximately 1 hour (1 second per listing).")

url_to_details = {}
for i, url in enumerate(voursa_urls):
    # Skip if we already have data for this URL
    url_df = df[df["url"] == url]
    if not url_df.empty and url_df[missing_cols].notna().any().any():
        # Already has some detail data, skip
        continue

    try:
        r = session.get(url, headers=DEFAULT_HEADERS, timeout=15)
        r.raise_for_status()
        url_to_details[url] = parse_voursa_detail_html(r.text)
    except Exception as e:
        url_to_details[url] = {}

    if (i + 1) % 50 == 0:
        print(f"Enriched {i + 1}/{len(voursa_urls)} detail pages...")
    time.sleep(1.0)

# Update DataFrame
for col in missing_cols:
    df[col] = df["url"].map(lambda u: url_to_details.get(u, {}).get(col))

print(f"\nEnriched {len(url_to_details)} listings with Overview & Details.")

# Save updated CSV
df.to_csv("real_estate_listings.csv", index=False)
print("Saved updated CSV with detail columns!")

# Show summary
print("\nSummary of detail columns:")
for col in detail_cols:
    non_null = df[col].notna().sum()
    print(f"  {col}: {non_null}/{len(df)} non-null values")
