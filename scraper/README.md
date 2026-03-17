# Real Estate Datasets Documentation

This repository contains three datasets related to real estate listings in Mauritania, specifically in Nouakchott and surrounding areas. Each dataset serves a different purpose in the data pipeline.

## Dataset Overview

| Dataset | Rows | Description | Status |
|---------|------|-------------|--------|
| `datasets/raw.csv` | 144 | Initial raw scraped data from wassit.info and lagence-mr.com | Raw |
| `datasets/raws.csv` | 4,337 | Merged comprehensive raw dataset from multiple sources | Raw |
| `datasets/real_estate_listings.csv` | 4,051 | Processed and cleaned dataset scraped from voursa.com | Processed |

---

## 1. raw.csv

**Purpose**: Initial raw scraped data from specific sources.

**Size**: 144 rows (143 data rows + 1 header)

**Sources**:
- `wassit.info` - Mauritanian classified ads website
- `lagence-mr.com` - Real estate agency website

**Columns**:
- `titre` - Property title/name
- `type_bien` - Property type (e.g., Appartement, Villa)
- `type_annonce` - Listing type (Location/Vente)
- `prix` - Price (in MRO - old Mauritanian ouguiya)
- `surface_m2` - Surface area in square meters
- `nb_chambres` - Number of bedrooms
- `nb_salons` - Number of living rooms
- `nb_sdb` - Number of bathrooms
- `quartier` - Neighborhood/district
- `ville` - City
- `description` - Full description (contains HTML and website boilerplate)
- `source` - Data source website
- `date_publication` - Publication date
- `caracteristiques` - Features/characteristics
- `url` - Source URL

**Characteristics**:
- Contains raw HTML content in descriptions
- Includes website navigation elements and boilerplate text
- Prices in old Mauritanian ouguiya (MRO)
- Mixed French and Arabic content
- Some entries may have incomplete or missing data

---

## 2. raws.csv

**Purpose**: Comprehensive merged raw dataset combining data from multiple sources.

**Size**: 4,337 rows (4,336 data rows + 1 header)

**Sources**:
- `voursa.com` - Primary source (majority of entries)
- `wassit.info` - Mauritanian classified ads
- `lagence-mr.com` - Real estate agency

**Columns**: Same structure as `raw.csv`
- `titre` - Property title/name
- `type_bien` - Property type
- `type_annonce` - Listing type (Location/Vente)
- `prix` - Price
- `surface_m2` - Surface area in square meters
- `nb_chambres` - Number of bedrooms
- `nb_salons` - Number of living rooms
- `nb_sdb` - Number of bathrooms
- `quartier` - Neighborhood/district
- `ville` - City
- `description` - Full description
- `source` - Data source website
- `date_publication` - Publication date
- `caracteristiques` - Features/characteristics
- `url` - Source URL

**Characteristics**:
- Contains all data from `raw.csv` plus additional scraped entries
- Largest raw dataset available
- Mix of property types: apartments, houses, land plots, duplexes, commercial properties
- Covers multiple neighborhoods in Nouakchott (Tevragh Zeina, Teyarett, Arafat, Dar Naim, etc.)
- Prices range from small amounts to millions of MRU/MRO
- Contains both French and Arabic text
- Raw format with HTML and website elements still present

**Note**: This file was created by merging `raw.csv` with additional scraped data.

---

## 3. real_estate_listings.csv

**Purpose**: Processed and cleaned dataset ready for analysis and machine learning.

**Size**: 4,051 rows (4,050 data rows + 1 header)

**Sources**: 
- **Scraped from `voursa.com`** - All listings in this dataset originate from voursa.com
- Processed and cleaned from raw scraped data

**Columns**:
- `raw_text` - Original raw text from the listing
- `price_mru` - Price in MRU (new Mauritanian ouguiya)
- `property_type` - Standardized property type (Land Plot, Residential Property, Duplex, Office, Store, etc.)
- `time_ago` - Time since publication (e.g., "4 hours ago", "2 months ago")
- `title_guess` - Extracted/guessed title
- `location` - Neighborhood/district name
- `source` - Data source website
- `url` - Source URL
- `lot_size` - Lot size in square meters
- `street_size` - Street size in meters
- `closest_landmark` - Nearest landmark
- `land_title` - Land title status (Yes/No)
- `detail_property_type` - Detailed property type classification
- `halls` - Number of halls/living rooms
- `balconies` - Number of balconies
- `bathrooms` - Number of bathrooms
- `rooms` - Number of rooms




---

## Data Flow

```
raw.csv (initial scrape from wassit.info & lagence-mr.com)
    ↓
real_estate_listings.csv (processed dataset scraped from voursa.com) 
    ↓
raws.csv (merged comprehensive raw data from multiple sources)
 
```




## Data Sources

### voursa.com
- Primary source for most listings
- Mauritanian real estate platform
- Contains listings in French and Arabic
- Includes property details, locations, and prices

### wassit.info
- Mauritanian classified ads website
- Contains various types of listings including real estate
- Older website with HTML-heavy content
- Prices in old Mauritanian ouguiya (MRO)

### lagence-mr.com
- Real estate agency website
- Professional listings with structured data
- Focus on high-end properties
- Prices in MRU

---

## Geographic Coverage

The datasets primarily cover **Nouakchott, Mauritania** and its neighborhoods:

- **Tevragh Zeina** - Popular residential area
- **Teyarett** - Central district
- **Arafat** - Residential neighborhood
- **Dar Naim** - Northern district
- **Toujounine** - Residential area
- **Ksar** - Historical district
- **Riyadh** - Residential area
- **Sebkha** - District
- **Nouadhibou** - Second largest city (some listings)

---

## Property Types

The datasets include various property types:

- **Residential Property** (Maison, Appartement, Villa)
- **Land Plot** (Terrain, Nيمرو)
- **Duplex** (Duplexe, ديبلكس)
- **Commercial** (Store, Office, Warehouse)
- **Other** (Various specialized properties)

---

## Notes

1. **Currency**: The datasets use both MRO (old Mauritanian ouguiya) and MRU (new Mauritanian ouguiya). The conversion rate is approximately 1 MRU = 10 MRO.

2. **Language**: Listings contain both French and Arabic text, reflecting the bilingual nature of Mauritania.

3. **Data Quality**: Raw datasets (`raw.csv` and `raws.csv`) contain HTML elements, website boilerplate, and may have incomplete entries. The processed dataset (`real_estate_listings.csv`) has been cleaned but may still contain some inconsistencies.

4. **Missing Values**: Raw datasets have many missing values, especially for numeric fields like `surface_m2`, `nb_chambres`, etc. The processed dataset has better coverage but may still have gaps.

5. **Updates**: `raws.csv` is the result of merging `raw.csv` with additional scraped data. `real_estate_listings.csv` is scraped exclusively from voursa.com and has been processed and cleaned for analysis.






---

## Project Structure

```
/home/moustapha/projectML/scraper-project/
├── datasets/                    # All CSV and data files
│   ├── raw.csv
│   ├── raws.csv
│   ├── real_estate_listings.csv
│   └── real_estate_listings.jsonl
├── scripts/                     # All Jupyter notebooks and Python scripts
│   ├── scraper.ipynb
│   ├── scraper_refactored.ipynb
│   ├── scraper_three_sites.ipynb
│   ├── real_estate_eda.ipynb
│   └── add_detail_columns.py
└── README.md                   # This file
```

---

*Last updated: Based on current dataset analysis*
