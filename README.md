# SCP Project - Net Congestion Analytics Platform

An interactive map platform for analyzing heat network potential and energy consumption of Dutch government (RVB) properties, supporting decisions to reduce net congestion through local heat source utilization.

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-username/SCP_Project.git
cd SCP_Project

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the script
python create_comprehensive_map.py
```

The script generates `comprehensive_energy_map.html` and opens it in your browser.

## Project Structure

```
SCP_Project/
├── create_comprehensive_map.py   # Main script - generates the interactive map
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── MANUAL.md                     # Detailed user manual (kaartlagen, scores, etc.)
├── comprehensive_energy_map.html # Generated output (after running script)
└── data/                         # Data directory (see Data Sources below)
    ├── Bouwwerken_netcongestie_data/   # RVB buildings shapefile
    ├── Country_data/                    # Netherlands boundary
    ├── defensie_data/                   # Defensie VKA GeoJSON files
    ├── Netherlands_shapefile/           # NL administrative boundary
    ├── tennet_data/                     # TenNet grid congestion data
    ├── TUD_data/                        # Energy consumption data (TU Delft)
    ├── warmte_data/                     # Heat source datasets (CSV, NetCDF)
    └── Warmte_net_data/                 # Heat network data (Excel, Shapefile)
```

## Features

### Map Layers
- **RVB Buildings** - Government properties with energy consumption analysis
- **Defensie VKA** - Military facility locations (Bovenregionaal & Locatiespecifiek)
- **Warmte Net Areas** - Existing heat network coverage polygons
- **Heat Sources** - MT Warmte, Datacenters, Industrial waste heat, Geothermal

### Analytics per Building
- **Warmte Score (0-100)** - Heat savings potential based on nearby sources (<1km)
- **Oordeel Verbruik** - Current vs. potential consumption rating (Groen/Oranje/Rood)
- **Op bestaand warmtenet** - Whether building is within existing heat network (Ja/Nee)
- **Besparing potential** - Estimated MW savings from local heat utilization

### Dashboard Panels
- **Top 10 Potentiële Groei** - Buildings with highest heat savings potential
- **Data Legend** - Visual guide to map markers and colors

## Data Sources

| Dataset | Source | Format |
|---------|--------|--------|
| RVB Buildings | Rijksvastgoedbedrijf | Shapefile |
| Energy Consumption | TU Delft (TUD) | Excel |
| MT Warmte | PBL Startanalyse 2024 | CSV |
| Datacenter Warmte | RVO | CSV |
| Condens Warmte | RVO | CSV |
| PDOK Restwarmte | PDOK WFS Service | GeoJSON |
| Geothermal | ThermoGIS | NetCDF |
| Warmtenetten | CBS Buurtkaart 2020 | Shapefile + Excel |
| Defensie VKA | Defensie | GeoJSON |

## Requirements

- Python 3.8+
- Dependencies (see requirements.txt):
  - folium >= 0.14.0
  - geopandas >= 0.13.0
  - pandas >= 2.0.0
  - numpy >= 1.24.0
  - requests >= 2.28.0
  - netCDF4 >= 1.6.0
  - openpyxl >= 3.1.0
  - shapely >= 2.0.0
  - matplotlib >= 3.7.0

## Configuration

The script uses relative paths from the project root. Ensure the `data/` directory structure matches the expected layout (see Project Structure above).

Key data files required:
- `data/Bouwwerken_netcongestie_data/Bouwwerken_netcongestie.shp`
- `data/TUD_data/TUD_Basislijst_Bekende_aansluitingen_(sept25).xlsx`
- `data/Warmte_net_data/Download-WarmteNetten-XLS.xlsx`
- `data/Warmte_net_data/Buurtkaart_2020_v3/*.shp`
- `data/warmte_data/*.csv`
- `data/warmte_data/OVERVIEW_potential_recoverable_heat.nc`

## Output

The script generates:
1. **comprehensive_energy_map.html** - Interactive Folium map with all layers
2. Console output showing data loading progress and summary statistics

## For Detailed Information

See [MANUAL.md](MANUAL.md) for:
- Detailed explanation of map layers
- Score calculation methodology
- Field definitions and interpretations
- Data source references

## License

Internal project - contact project maintainers for usage terms.
