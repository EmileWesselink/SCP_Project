# SCP Project - Net Congestion Analytics Platform

An interactive map platform for analyzing heat network potential and energy consumption of Dutch government (RVB) properties, supporting decisions to reduce net congestion through local heat source utilization.

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-username/SCP_Project.git
cd SCP_Project

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the script
python create_comprehensive_map.py
```

The script generates `comprehensive_energy_map.html` and opens it in your browser.

---

## Project Structure

```
SCP_Project/
├── create_comprehensive_map.py   # Main script - generates the interactive map
├── requirements.txt              # Python dependencies
├── README.md                     # This file (technical documentation)
├── MANUAL.md                     # User manual (non-technical)
├── comprehensive_energy_map.html # Generated output (after running script)
└── data/                         # Data directory
    ├── Bouwwerken_netcongestie_data/   # RVB buildings shapefile
    ├── Country_data/                    # Netherlands boundary
    ├── defensie_data/                   # Defensie VKA GeoJSON files
    ├── Netherlands_shapefile/           # NL administrative boundary
    ├── tennet_data/                     # TenNet grid congestion data
    ├── TUD_data/                        # Energy consumption data (TU Delft)
    ├── warmte_data/                     # Heat source datasets (CSV, NetCDF)
    └── Warmte_net_data/                 # Heat network data (Excel, Shapefile)
```

---

## Features

### Map Layers
- **RVB Buildings** - Government properties with energy consumption analysis
- **Defensie VKA** - Military facility locations (Bovenregionaal & Locatiespecifiek)
- **Heat Network Areas** - Existing heat network coverage polygons
- **Heat Sources** - MT Warmte, Datacenters, Industrial waste heat, Geothermal

### Analytics per Building
- **Heat Score (0-100)** - Heat savings potential based on nearby sources (<1km)
- **Consumption Rating** - Current vs. potential consumption assessment (Green/Orange/Red)
- **On existing heat network** - Whether the building is within an existing heat network (Yes/No)
- **Savings potential** - Estimated MW savings through local heat utilization

### Dashboard Panels
- **Top 10 Potential Growth** - Buildings with highest heat savings potential
- **Data Legend** - Visual guide for map markers and colors

---

## Data Sources

| Dataset | Source | Format | Location |
|---------|--------|--------|----------|
| RVB Buildings | Rijksvastgoedbedrijf | Shapefile | `data/Bouwwerken_netcongestie_data/` |
| Energy Consumption | TU Delft (TUD) | Excel | `data/TUD_data/` |
| MT Warmte | PBL Startanalyse 2024 | CSV | `data/warmte_data/` |
| Datacenter Heat | RVO | CSV | `data/warmte_data/` |
| PDOK Waste Heat | PDOK WFS Service | GeoJSON (live) | Fetched via API |
| Geothermal | ThermoGIS | NetCDF | `data/warmte_data/` |
| Heat Networks | CBS Buurtkaart 2020 | Shapefile + Excel | `data/Warmte_net_data/` |
| Defensie VKA | Defensie | GeoJSON | `data/defensie_data/` |
| Netherlands Boundary | GADM | Shapefile | `data/Country_data/` |

---

## Requirements

- Python 3.8+
- Dependencies (see `requirements.txt`):

```
folium>=0.14.0
geopandas>=0.13.0
pandas>=2.0.0
numpy>=1.24.0
requests>=2.28.0
netCDF4>=1.6.0
openpyxl>=3.1.0
shapely>=2.0.0
matplotlib>=3.7.0
pyproj>=3.5.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Configuration

The script uses relative paths from the project root. Required data files:

```
data/Bouwwerken_netcongestie_data/Bouwwerken_netcongestie.shp
data/TUD_data/TUD_Basislijst_Bekende_aansluitingen_(sept25).xlsx
data/Warmte_net_data/Download-WarmteNetten-XLS.xlsx
data/Warmte_net_data/Buurtkaart_2020_v3/*.shp
data/warmte_data/*.csv
data/warmte_data/OVERVIEW_potential_recoverable_heat.nc
```

---

## Adding New Data Layers

This section explains how to add new data sources to the map.

### Overview

The map uses [Folium](https://python-visualization.github.io/folium/) for visualization. Each layer is a `FeatureGroup` that can be toggled on/off via the layer menu.

### Step 1: Prepare Your Data

Your data must be in one of these formats:
- **CSV** with coordinates (X/Y or lat/lon columns)
- **Shapefile** (.shp with .dbf, .shx, .prj)
- **GeoJSON** (.geojson)
- **Excel** (.xlsx) with coordinate columns
- **NetCDF** (.nc) for raster/grid data

Place the data file in the appropriate `data/` subdirectory.

### Step 2: Load the Data

Add data loading code near the top of `create_comprehensive_map.py` (around line 400-600 where other data is loaded):

#### For CSV files:
```python
# Load your CSV data
my_data_file = 'data/warmte_data/Your_Data_File.csv'
if os.path.exists(my_data_file):
    my_df = pd.read_csv(my_data_file)
    print(f"Loaded: {len(my_df)} records from your data")
```

#### For Shapefiles:
```python
# Load shapefile
my_shapefile = 'data/your_folder/your_data.shp'
if os.path.exists(my_shapefile):
    my_gdf = gpd.read_file(my_shapefile)
    my_gdf = my_gdf.to_crs(epsg=4326)  # Convert to WGS84 lat/lon
    print(f"Loaded: {len(my_gdf)} features from shapefile")
```

#### For GeoJSON:
```python
# Load GeoJSON
my_geojson = 'data/your_folder/your_data.geojson'
if os.path.exists(my_geojson):
    my_gdf = gpd.read_file(my_geojson)
    print(f"Loaded: {len(my_gdf)} features from GeoJSON")
```

### Step 3: Convert Coordinates (if needed)

If your data uses Dutch RD coordinates (EPSG:28992), convert to WGS84:

```python
from pyproj import Transformer

# Create transformer from RD to WGS84
transformer = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)

# For DataFrame with X, Y columns
my_df['lon'], my_df['lat'] = transformer.transform(
    my_df['X'].values,
    my_df['Y'].values
)
```

### Step 4: Create a Feature Group

Add a new FeatureGroup for your layer (around line 1200-1400):

```python
# ============ YOUR NEW LAYER ============
print("Adding Your New Layer...")
my_layer_group = folium.FeatureGroup(name='🔶 Your Layer Name', show=False)

for idx, row in my_gdf.iterrows():
    # Create popup HTML
    popup_html = f"""
    <div style="font-family: Arial; width: 280px;">
        <h4 style="color: #FF6600; margin-bottom: 10px;">
            🔶 Your Data Type
        </h4>
        <table style="width: 100%; font-size: 12px;">
            <tr><td><b>Name:</b></td><td>{row.get('Name', 'N/A')}</td></tr>
            <tr><td><b>Value:</b></td><td>{row.get('Value', 'N/A')}</td></tr>
        </table>
    </div>
    """

    # Add marker to the layer
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=6,
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=f"Your Data: {row.get('Name', 'N/A')}",
        color='#FF6600',      # Border color
        fillColor='#FF9933',  # Fill color
        fillOpacity=0.7,
        weight=2
    ).add_to(my_layer_group)

my_layer_group.add_to(m)
print(f"  Added: {len(my_gdf)} markers to Your Layer")
```

### Step 5: Include in Score Calculation (Optional)

If your data should affect the heat score, modify the `calculate_location_score()` function:

1. Add to `all_warmte_sources` list during data loading:
```python
all_warmte_sources.append({
    'lat': row.geometry.y,
    'lon': row.geometry.x,
    'type': 'Your Data Type',
    'name': row.get('Name', 'N/A'),
    'color': '#FF6600',
    'your_value': row.get('Value', 0),
    'power_display': f"{row.get('Value', 0):.1f} MW"
})
```

2. Add to score_breakdown in `calculate_location_score()`:
```python
score_breakdown = {
    'mt_warmte_mwth': 0.0,
    'datacenter_vermogen': 0.0,
    'your_data_value': 0.0,  # Add this
    'geothermie_heat': 0.0
}
```

3. Add processing logic:
```python
elif source['type'] == 'Your Data Type':
    score_breakdown['your_data_value'] += source.get('your_value', 0) or 0
```

4. Include in raw_score calculation:
```python
raw_score = (
    score_breakdown['mt_warmte_mwth'] * 1.0 +
    score_breakdown['datacenter_vermogen'] * 1.0 +
    score_breakdown['your_data_value'] * 0.5 +  # Adjust weight as needed
    score_breakdown['geothermie_heat'] * 0.01
)
```

### Step 6: Update Documentation

Add your new layer to:
- `MANUAL.md` - User documentation
- `README.md` - Data sources table

### Example: Adding a Polygon Layer

For polygon data (such as zones or areas):

```python
# ============ YOUR POLYGON LAYER ============
my_polygon_group = folium.FeatureGroup(name='📍 Your Zones', show=False)

for idx, row in my_zones_gdf.iterrows():
    # Style the polygon
    style = {
        'fillColor': '#FF6600',
        'color': '#CC5500',
        'weight': 2,
        'fillOpacity': 0.3
    }

    popup_html = f"<b>{row.get('ZoneName', 'Zone')}</b><br>Area: {row.get('Area', 'N/A')}"

    folium.GeoJson(
        row.geometry.__geo_interface__,
        style_function=lambda x, s=style: s,
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=row.get('ZoneName', 'Zone')
    ).add_to(my_polygon_group)

my_polygon_group.add_to(m)
```

### Example: Adding a Heatmap Layer

For density/intensity visualization:

```python
from folium.plugins import HeatMap

# Prepare heat data: list of [lat, lon, intensity]
heat_data = [[row.geometry.y, row.geometry.x, row.get('intensity', 1)]
             for idx, row in my_gdf.iterrows()]

# Create heatmap
HeatMap(
    heat_data,
    min_opacity=0.3,
    max_zoom=13,
    radius=15,
    blur=10,
    gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}
).add_to(folium.FeatureGroup(name='🔥 Heat Intensity').add_to(m))
```

---

## Disabled Layers

Some layers are commented out in the code and can be re-enabled:

### Condens Warmte (Cooling Processes)

This layer is currently disabled. To re-enable:

1. Search for `CONDENS WARMTE DATA COLLECTION` in `create_comprehensive_map.py`
2. Uncomment the data loading block (lines ~527-612)
3. Search for `CONDENS WARMTE LAYER`
4. Uncomment the layer creation block (lines ~1249-1361)
5. Uncomment the score calculation lines marked with `# DISABLED`

---

## Score Calculation

### Formula

```
raw_score = (MT_Warmte_MWth × 1.0) + (Datacenter_Vermogen × 1.0) + (Geothermie_Heat × 0.01)
```

### Normalization

```
normalized_score = 30 × log₁₀(raw_score + 1) + 10
```

This logarithmic scaling provides:
- 1 MW ≈ 30 points
- 10 MW ≈ 60 points
- 100 MW ≈ 90 points

### Score Interpretation

| Score | Label | Meaning |
|------:|-------|---------|
| 80+ | Excellent | Excellent heat potential |
| 60-79 | Good | Good potential |
| 40-59 | Moderate | Moderate potential |
| 20-39 | Limited | Limited potential |
| <20 | Minimal | Minimal potential |

### Savings Calculation

```
savings = min(raw_score, total_consumption × 0.5)
```

Savings are capped at 50% of the building's total energy consumption.

---

## Output

The script generates:
1. **comprehensive_energy_map.html** - Interactive Folium map with all layers
2. Console output with data loading progress and summary statistics

---

## Troubleshooting

### Common Issues

**"FileNotFoundError: data/..."**
- Verify that the data structure matches the expected layout
- Check that file paths are relative to the project root

**"CRS transformation error"**
- Install pyproj: `pip install pyproj`
- Verify that the source CRS is correctly specified

**"Memory error with large datasets"**
- Reduce the number of points by sampling
- Use simpler geometries (centroids instead of polygons)

**"Map doesn't load in browser"**
- Try a different browser (Chrome/Firefox recommended)
- Check if the HTML file is too large (>50MB may cause issues)

### Performance Tips

- Set `show=False` for heavy layers to improve initial load time
- Use `simplify()` on complex geometries
- Limit the number of markers per layer to <5000

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test the map generation
5. Submit a pull request

---

## Non-Technical Information

See [MANUAL.md](MANUAL.md) for:
- How to use the map
- What the symbols and colors mean
- How to interpret the scores
- Frequently asked questions

## License

Unless stated otherwise, this project is licensed under the CC BY-NC-SA 4.0 License. See the LICENSE file for details.
