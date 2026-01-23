# -*- coding: utf-8 -*-
"""
SCP Project - Comprehensive Energy Infrastructure Map

This script creates an interactive map visualizing:
- RVB (Rijksvastgoedbedrijf) building locations with energy consumption data
- Warmte (heat) sources: MT Warmte, Datacenters, Industrial waste heat, Geothermal
- Warmtenetten (heat network) coverage areas
- Defensie VKA locations with heat potential analysis

Features:
- Heat savings potential score (0-100) for each building
- Top 10 buildings with highest growth potential
- "Op bestaand warmtenet" indicator for each RVB building
- Interactive popups with detailed analytics

Usage:
    python create_comprehensive_map.py

Output:
    comprehensive_energy_map.html - Interactive map file

Author: SCP Project Team
"""
# ============================================================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Standard library imports
import os
import glob
import json
import base64
import webbrowser

# Data processing
import pandas as pd
import numpy as np
import geopandas as gpd
import netCDF4
import requests

# Visualization
import folium
from folium import plugins
from folium.plugins import HeatMap, MiniMap, Fullscreen, MeasureControl
from matplotlib import colors

# Geometry operations
from shapely.ops import unary_union

# ============ GEOCODING FUNCTION FOR DUTCH PLACES ============
def geocode_plaats(plaats_name, cache={}):
    """
    Geocode a Dutch place name to RD New coordinates using PDOK Locatieserver.
    Uses caching to avoid repeated API calls.
    Returns (x, y) in EPSG:28992 or (None, None) if not found.
    """
    if plaats_name in cache:
        return cache[plaats_name]

    try:
        # Clean up the place name
        clean_name = plaats_name.strip().upper()

        # PDOK Locatieserver API
        url = f"https://api.pdok.nl/bzk/locatieserver/search/v3_1/free?q={clean_name}&fq=type:woonplaats&rows=1"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            data = response.json()
            if data.get('response', {}).get('docs'):
                doc = data['response']['docs'][0]
                # Get centroid coordinates (in RD New)
                if 'centroide_rd' in doc:
                    # Format: "POINT(x y)"
                    centroid = doc['centroide_rd']
                    coords = centroid.replace('POINT(', '').replace(')', '').split()
                    x, y = float(coords[0]), float(coords[1])
                    cache[plaats_name] = (x, y)
                    return (x, y)

        cache[plaats_name] = (None, None)
        return (None, None)
    except Exception as e:
        cache[plaats_name] = (None, None)
        return (None, None)

# Pre-populate geocode cache with common Dutch cities to reduce API calls
geocode_cache = {}

# ============ HELPER FUNCTIONS FOR SCORE CALCULATION ============
def parse_vermogen_range(vermogen_str):
    """Parse VERMOGEN range string to get average numeric value in MW."""
    if pd.isna(vermogen_str) or vermogen_str == 'onbekend':
        return 0.0
    try:
        vermogen_str = str(vermogen_str).replace(',', '.')
        if vermogen_str.startswith('<'):
            return float(vermogen_str[1:]) / 2  # Take half of upper bound
        elif vermogen_str.startswith('>'):
            return float(vermogen_str[1:])  # Take lower bound
        elif '-' in vermogen_str:
            parts = vermogen_str.split('-')
            return (float(parts[0]) + float(parts[1])) / 2  # Average
        else:
            return float(vermogen_str)
    except:
        return 0.0

def parse_temp_range(temp_str):
    """Parse RESTW_TEMP range string to get average numeric value."""
    if pd.isna(temp_str) or temp_str == 'onbekend':
        return 0.0
    try:
        temp_str = str(temp_str)
        if '-' in temp_str:
            parts = temp_str.split('-')
            return (float(parts[0]) + float(parts[1])) / 2
        else:
            return float(temp_str)
    except:
        return 0.0

def haversine_km(lon1, lat1, lon2, lat2):
    """Calculate haversine distance in km between two points."""
    from math import radians, cos, sin, asin, sqrt
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c

def calculate_warmte_score(lat, lon, warmte_sources, geothermie_gdf=None, include_geothermie=False):
    """
    Calculate the warmte besparing score for a location.

    Components (all within 1km):
    - MT Warmte: MWth (thermal power in MW)
    - Datacenter: VERMOGEN where RESTW_TEMP > 60°C
    - Condens Warmte: TJ_MTWarmte (TJ of MT heat via heat pump)
    - Geothermie: heat value at location (only for Defensie)

    Returns: (raw_score, score_breakdown)
    """
    score_breakdown = {
        'mt_warmte_mwth': 0.0,
        'datacenter_vermogen': 0.0,
        'condens_tj_mt': 0.0,
        'geothermie_heat': 0.0
    }

    for source in warmte_sources:
        dist = haversine_km(lon, lat, source['lon'], source['lat'])
        if dist <= 1.0:  # Within 1km
            if source['type'] == 'MT Warmte':
                score_breakdown['mt_warmte_mwth'] += source.get('MWth', 0) or 0
            elif source['type'] == 'Datacenter':
                # Only count if temperature > 60°C
                temp = source.get('RESTW_TEMP_numeric', 0) or 0
                if temp > 60:
                    score_breakdown['datacenter_vermogen'] += source.get('VERMOGEN_numeric', 0) or 0
            elif source['type'] == 'Condens Warmte':
                score_breakdown['condens_tj_mt'] += source.get('TJ_MTWarmte', 0) or 0

    # Add geothermie for Defensie locations
    if include_geothermie and geothermie_gdf is not None and len(geothermie_gdf) > 0:
        # Find nearest geothermie point
        min_dist = float('inf')
        nearest_heat = 0
        for _, geo_row in geothermie_gdf.iterrows():
            dist = haversine_km(lon, lat, geo_row.geometry.x, geo_row.geometry.y)
            if dist < min_dist:
                min_dist = dist
                nearest_heat = geo_row.get('heat', 0) or 0
        # Only include if within reasonable distance (5km for geothermie)
        if min_dist <= 5.0:
            score_breakdown['geothermie_heat'] = nearest_heat

    # Calculate raw score (weighted sum)
    # Convert all to common unit (approximate energy potential in MW-equivalent)
    raw_score = (
        score_breakdown['mt_warmte_mwth'] * 1.0 +  # MW thermal
        score_breakdown['datacenter_vermogen'] * 1.0 +  # MW
        score_breakdown['condens_tj_mt'] * 0.1 +  # TJ -> approximate MW equivalent
        score_breakdown['geothermie_heat'] * 0.01  # Scale down geothermie
    )

    return raw_score, score_breakdown

def normalize_score(raw_score, max_score=100):
    """
    Normalize raw score to 0-100 scale using logarithmic scaling.
    This makes the score more interpretable across different magnitudes.

    Interpretation:
    - 0-20: Zeer Laag (nauwelijks warmtebronnen beschikbaar)
    - 20-40: Laag (beperkte warmtebronnen)
    - 40-60: Gemiddeld (redelijke warmtepotentie)
    - 60-80: Hoog (goede warmtepotentie)
    - 80-100: Zeer Hoog (uitstekende warmtepotentie)
    """
    if raw_score <= 0:
        return 0

    # Use logarithmic scaling: score of 1 MW = ~30, 10 MW = ~60, 100 MW = ~90
    import math
    # Log scale with base adjustment for reasonable distribution
    normalized = min(100, max(0, 30 * math.log10(raw_score + 1) + 10))
    return normalized

def get_score_interpretation(normalized_score):
    """
    Get textual interpretation and color for the normalized score.
    Returns: (label, color, description)
    """
    if normalized_score >= 80:
        return ("Zeer Hoog", "#1B5E20", "Uitstekende warmtepotentie - veel nabije bronnen beschikbaar")
    elif normalized_score >= 60:
        return ("Hoog", "#4CAF50", "Goede warmtepotentie - meerdere bruikbare bronnen")
    elif normalized_score >= 40:
        return ("Gemiddeld", "#FF9800", "Redelijke warmtepotentie - enkele bronnen beschikbaar")
    elif normalized_score >= 20:
        return ("Laag", "#FF5722", "Beperkte warmtepotentie - weinig nabije bronnen")
    else:
        return ("Zeer Laag", "#B71C1C", "Nauwelijks warmtebronnen binnen 1km beschikbaar")

def create_score_gauge_html(normalized_score, label, color):
    """Create a visual gauge/progress bar for the score."""
    return f"""
    <div style="margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
            <span style="font-size: 32px; font-weight: bold; color: {color};">{normalized_score:.0f}</span>
            <span style="background: {color}; color: white; padding: 4px 12px; border-radius: 12px; font-weight: bold; font-size: 12px;">{label}</span>
        </div>
        <div style="background: #e0e0e0; border-radius: 10px; height: 12px; overflow: hidden;">
            <div style="background: linear-gradient(90deg, #B71C1C 0%, #FF5722 25%, #FF9800 50%, #4CAF50 75%, #1B5E20 100%); width: {normalized_score}%; height: 100%; border-radius: 10px; transition: width 0.3s;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 9px; color: #666; margin-top: 2px;">
            <span>0</span>
            <span>Zeer Laag</span>
            <span>Laag</span>
            <span>Gemiddeld</span>
            <span>Hoog</span>
            <span>Zeer Hoog</span>
            <span>100</span>
        </div>
    </div>
    """

# Global variables for score normalization (will be set after collecting all scores)
all_raw_scores = []


print("=" * 80)
print("LOADING ALL DATA SOURCES...")
print("=" * 80)

# ============ 1. LOAD RVB BUILDINGS ============
print("\n[1/5] Loading RVB Buildings...")
gdf = gpd.read_file("data/Bouwwerken_netcongestie_data/Bouwwerken_netcongestie.shp")
TUD_basislijst = pd.read_excel("data/TUD_data/TUD_Basislijst_Bekende_aansluitingen_(sept25).xlsx", sheet_name = "Gefilterde data", header=0)
merged = gdf.merge(TUD_basislijst, on="EAN", how="inner")
gdf_wgs84 = merged.to_crs(epsg=4326)
gdf_projected = gdf.to_crs(epsg=28992)
gdf_projected["centroid"] = gdf_projected.geometry.centroid
gdf_wgs84["centroid"] = gdf_projected["centroid"].to_crs(epsg=4326)
rvb_points = gdf_wgs84.copy()
rvb_points["geometry"] = rvb_points["centroid"]
rvb_points["energy_proxy"] = rvb_points["Shape_Area"]
min_area = rvb_points["energy_proxy"].min()
max_area = rvb_points["energy_proxy"].max()
rvb_points["radius"] = 4 + (rvb_points["energy_proxy"] - min_area) / (max_area - min_area) * 14
rvb_points['vermogen/capacity']= rvb_points['Max vermogen verbruik'] / rvb_points['Toekomstige contractcapaciteit'] * 100
rvb_points['WP aanwezig'] = rvb_points['WP vermogen'].apply(lambda x: 'Ja' if x > 0 else 'Nee')


# Map judgement to colors
oordeel_color_map = {
    "Groen":  "#4CAF50",
    "Oranje": "#FF9800",
    "Rood":   "#F44336",
}

rvb_points["marker_color"] = (
    rvb_points["Oordeel verbruik"]
    .map(oordeel_color_map)
    .fillna("#1a5490")   # fallback if empty/unknown
)

print(f"✓ Loaded {len(rvb_points)} RVB buildings")

# ============ 2. LOAD DEFENSIE VKA DATA ============
print("\n[2/5] Loading Defensie VKA data...")
defensie_geojsons = glob.glob("data/defensie_data/20250827_export_VKAs/**/*.geojson", recursive=True)
print(f"  Found {len(defensie_geojsons)} Defensie GeoJSON files")

bovenregionaal_files = [f for f in defensie_geojsons if "Bovenregionaal VKA" in f]
locatiespecifiek_files = [f for f in defensie_geojsons if "Locatiespecifiek VKA" in f]

print(f"  - Bovenregionaal: {len(bovenregionaal_files)} files")
print(f"  - Locatiespecifiek: {len(locatiespecifiek_files)} files")

# ============ 3. LOAD TENNET DATA ============
print("\n[3/5] Loading TenNet data...")
tennet_data = {}
tennet_files = {
    'congestie': 'data/tennet_data/tennetcongestie.csv',
    'pc6': 'data/tennet_data/congestie_pc6.csv',
    'projecten': 'data/tennet_data/projecten.csv',
    'voedingsgebieden': 'data/tennet_data/voedingsgebieden.csv',
    'tennetgebieden': 'data/tennet_data/tennetgebieden.csv'
}

for key, filepath in tennet_files.items():
    if os.path.exists(filepath):
        try:
            if key in ['congestie', 'pc6']:
                df = pd.read_csv(filepath, sep=';', on_bad_lines='skip', encoding='utf-8')
            else:
                df = pd.read_csv(filepath, on_bad_lines='skip', encoding='utf-8')
            tennet_data[key] = df
            print(f"  ✓ {key}: {len(df)} records")
        except Exception as e:
            print(f"  ○ {key}: Could not load - {str(e)[:60]}")
    else:
        print(f"  ○ {key}: File not found")

# ============ 4. LOAD WARMTE DATA ============
print("\n[4/5] Loading Warmte (heat) data...")
warmte_data = {}

# Load NetCDF warmte grid
nc_fp = "data/warmte_data/OVERVIEW_potential_recoverable_heat.nc"

try:
    nc = netCDF4.Dataset(nc_fp)

    # Read variables (names taken from the .nc you showed earlier)
    data = nc.variables["data"][:]    # 2D array [y, x]
    x = nc.variables["x"][:]          # 1D array (x coordinates in RD New)
    y = nc.variables["y"][:]          # 1D array (y coordinates in RD New)


    # Build a point cloud (one point per non-NaN grid cell)
    X_grid, Y_grid = np.meshgrid(x, y)          # shape (ny, nx)
    flat_df = pd.DataFrame({
        "X": X_grid.ravel(),
        "Y": Y_grid.ravel(),
        "heat": data.ravel()
    })

    # Drop NaNs
    flat_df = flat_df.dropna(subset=["heat"])

    # Turn into GeoDataFrame in RD New (EPSG:28992), then to WGS84 (EPSG:4326)
    gdf_heat = gpd.GeoDataFrame(
        flat_df,
        geometry=gpd.points_from_xy(flat_df["X"], flat_df["Y"]),
        crs="EPSG:28992"
    ).to_crs(epsg=4326)

    # Store in warmte_data dict so the map code can access it
    warmte_data["OVERVIEW_potential_recoverable_heat.nc"] = gdf_heat

    print(f"  ✓ Loaded NetCDF warmte grid: {len(gdf_heat)} non-empty cells")

except Exception as e:
    print(f"  ○ NetCDF warmte grid not loaded - {str(e)[:80]}")

# --- Existing CSV warmte files loading ---
warmte_files = glob.glob("data/warmte_data/*.csv")
print(f"  Found {len(warmte_files)} warmte CSV files")

for filepath in warmte_files:
    filename = os.path.basename(filepath)
    try:
        df = pd.read_csv(filepath, sep=';', on_bad_lines='skip')
        warmte_data[filename] = df
        print(f"  ✓ {filename}: {len(df)} records")
    except Exception as e:
        print(f"  ○ {filename}: Could not load - {str(e)[:50]}")



# ============ 5. LOAD NETHERLANDS BOUNDARY ============
print("\n[5/5] Loading Netherlands boundary...")
nl = gpd.read_file("data/Netherlands_shapefile/nl_1km.shp")
nl_wgs84 = nl.to_crs(epsg=4326)
nl_dissolved = nl_wgs84.dissolve()
print(f"✓ Netherlands boundary loaded")

center_lat = rvb_points.geometry.y.mean()
center_lon = rvb_points.geometry.x.mean()

print("\n" + "=" * 80)
print("CREATING INTERACTIVE MAP...")
print("=" * 80)
#====================== LOAD Warmte Net Data and Merge with Buurtkaart ======================
warmte_net = pd.read_excel("data/Warmte_net_data/Download-WarmteNetten-XLS.xlsx", header=0)

# Define the folder path
folder_path = "data/Warmte_net_data/Buurtkaart_2020_v3"
# List all files in the folder
files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.shp')]

# Read all shapefiles into a list of GeoDataFrames
buurtkaarten = [gpd.read_file(file) for file in files]

# Display the first few rows of each GeoDataFrame
for i, buurtkaart in enumerate(buurtkaarten):
    buurtkaarten[i] = buurtkaart[['BU_CODE', 'geometry']]   
buurtkaart_gdf = gpd.GeoDataFrame(pd.concat(buurtkaarten, ignore_index=True), crs=buurtkaarten[0].crs)
buurt_warmte_net = buurtkaart_gdf.merge(warmte_net, on='BU_CODE', how='inner')

# Convert to WGS84 for Folium compatibility
buurt_warmte_net = buurt_warmte_net.to_crs(epsg=4326)
print(f"  Loaded {len(buurt_warmte_net)} warmtenet buurt areas")

# ============ CHECK RVB BUILDINGS ON WARMTE NET ============
print("Checking which RVB buildings are on existing warmte net...")

# Create a single unified geometry of all warmte net areas for faster spatial check
warmte_net_union = unary_union(buurt_warmte_net.geometry)

# Check if each RVB point is within the warmte net
def check_op_warmtenet(point_geom):
    """Check if a point is within the warmte net coverage area."""
    try:
        return 'Ja' if warmte_net_union.contains(point_geom) else 'Nee'
    except:
        return 'Nee'

rvb_points['Op bestaand warmtenet'] = rvb_points.geometry.apply(check_op_warmtenet)
warmtenet_count = (rvb_points['Op bestaand warmtenet'] == 'Ja').sum()
print(f"  RVB buildings on existing warmte net: {warmtenet_count} / {len(rvb_points)}")

# ============ CREATE BASE MAP ============
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=5,
    tiles=None,
    control_scale=False,
    prefer_canvas=True
)

folium.TileLayer('CartoDB positron', name='Light Map', attr='CartoDB').add_to(m)
folium.TileLayer('OpenStreetMap', name='Street Map', attr='OpenStreetMap').add_to(m)
folium.TileLayer('CartoDB dark_matter', name='Dark Map', attr='CartoDB').add_to(m)

# ============ NETHERLANDS BOUNDARY ============
boundary_group = folium.FeatureGroup(name='🗺️ Netherlands Boundary', show=True)
folium.GeoJson(
    nl_dissolved,
    style_function=lambda x: {
        'fillColor': 'transparent',
        'color': '#1a5490',
        'weight': 3,
        'fillOpacity': 0
    }
).add_to(boundary_group)
boundary_group.add_to(m)

# ============ COLLECT ALL WARMTE SOURCES FOR ANALYTICS (must be before RVB) ============
print("Collecting all warmte sources for analytics...")
all_warmte_sources = []

# MT Warmte - include MWth (thermal power) for score calculation
mt_warmte_file = 'Download-MT-Warmtebronnen startanalyse  (2024)-CSV.csv'
if mt_warmte_file in warmte_data:
    mt_df = warmte_data[mt_warmte_file]
    if 'X' in mt_df.columns and 'Y' in mt_df.columns:
        mt_with_coords = mt_df.dropna(subset=['X', 'Y'])
        if len(mt_with_coords) > 0:
            gdf_mt = gpd.GeoDataFrame(mt_with_coords, geometry=gpd.points_from_xy(mt_with_coords['X'], mt_with_coords['Y']), crs='EPSG:28992').to_crs(epsg=4326)
            for idx, row in gdf_mt.iterrows():
                # Parse MWth value (thermal power in MW)
                mwth_val = row.get('MWth', 0)
                try:
                    mwth_val = float(mwth_val) if pd.notna(mwth_val) else 0.0
                except:
                    mwth_val = 0.0

                all_warmte_sources.append({
                    'lat': row.geometry.y,
                    'lon': row.geometry.x,
                    'type': 'MT Warmte',
                    'name': row.get('BronNaam', 'N/A'),
                    'gemeente': row.get('Gemeente', 'N/A'),
                    'color': '#1E90FF',
                    'MWth': mwth_val,
                    'power_display': f"{mwth_val:.1f} MW" if mwth_val > 0 else "N/A"
                })



# Datacenter - include VERMOGEN and RESTW_TEMP for score calculation
datacenter_file = 'Download-LT DataCentraWarmte-CSV.csv'
if datacenter_file in warmte_data:
    dc_df = warmte_data[datacenter_file]
    if 'X' in dc_df.columns and 'Y' in dc_df.columns:
        dc_with_coords = dc_df.dropna(subset=['X', 'Y'])
        if len(dc_with_coords) > 0:
            gdf_dc = gpd.GeoDataFrame(dc_with_coords, geometry=gpd.points_from_xy(dc_with_coords['X'], dc_with_coords['Y']), crs='EPSG:28992').to_crs(epsg=4326)
            for idx, row in gdf_dc.iterrows():
                # Parse VERMOGEN and RESTW_TEMP
                vermogen_numeric = parse_vermogen_range(row.get('VERMOGEN', '0'))
                temp_numeric = parse_temp_range(row.get('RESTW_TEMP', '0'))

                all_warmte_sources.append({
                    'lat': row.geometry.y,
                    'lon': row.geometry.x,
                    'type': 'Datacenter',
                    'name': row.get('BEDRIJF', 'N/A'),
                    'gemeente': row.get('WOONPLAATS', 'N/A'),
                    'color': '#9370DB',
                    'VERMOGEN_numeric': vermogen_numeric,
                    'RESTW_TEMP_numeric': temp_numeric,
                    'VERMOGEN_raw': row.get('VERMOGEN', 'N/A'),
                    'RESTW_TEMP_raw': row.get('RESTW_TEMP', 'N/A'),
                    'power_display': f"{vermogen_numeric:.1f} MW ({row.get('RESTW_TEMP', 'N/A')}°C)"
                })

# ============ CONDENS WARMTE DATA COLLECTION (COMMENTED OUT) ============
# To re-enable: uncomment the block below
# --- START CONDENS WARMTE DATA ---
# condens_file = 'Download-LT CondensWarmte uit Koelprocessen-CSV.csv'
# if condens_file in warmte_data:
#     cw_df = warmte_data[condens_file]
#     # This file has 'Plaats' instead of X,Y - need to geocode
#     if 'Plaats' in cw_df.columns:
#         print("  Geocoding Condens Warmte locations (this may take a moment)...")
#         cw_with_plaats = cw_df.dropna(subset=['Plaats'])
#         geocoded_count = 0
#         geocoded_data = []
#
#         # Group by Plaats to reduce API calls
#         unique_places = cw_with_plaats['Plaats'].unique()
#         plaats_coords = {}
#         for plaats in unique_places:
#             x, y = geocode_plaats(plaats, geocode_cache)
#             if x is not None and y is not None:
#                 plaats_coords[plaats] = (x, y)
#
#         # Now process all rows with geocoded coordinates - include TJ_MTWarmte for score
#         for idx, row in cw_with_plaats.iterrows():
#             plaats = row['Plaats']
#             if plaats in plaats_coords:
#                 x, y = plaats_coords[plaats]
#                 # Parse TJ_MTWarmte value
#                 tj_mt_val = row.get('TJ_MTWarmte', 0)
#                 try:
#                     tj_mt_val = float(tj_mt_val) if pd.notna(tj_mt_val) else 0.0
#                 except:
#                     tj_mt_val = 0.0
#
#                 geocoded_data.append({
#                     'X': x,
#                     'Y': y,
#                     'Naam': row.get('Naam', 'N/A'),
#                     'Plaats': plaats,
#                     'TJ_CondWarmte': row.get('TJ_CondWarmte', 'N/A'),
#                     'TJ_MTWarmte': tj_mt_val,
#                     'SBINaam': row.get('SBINaam', 'N/A')
#                 })
#                 geocoded_count += 1
#
#         if geocoded_data:
#             geocoded_df = pd.DataFrame(geocoded_data)
#             gdf_cw = gpd.GeoDataFrame(
#                 geocoded_df,
#                 geometry=gpd.points_from_xy(geocoded_df['X'], geocoded_df['Y']),
#                 crs='EPSG:28992'
#             ).to_crs(epsg=4326)
#             for idx, row in gdf_cw.iterrows():
#                 tj_mt_val = row.get('TJ_MTWarmte', 0) or 0
#                 all_warmte_sources.append({
#                     'lat': row.geometry.y,
#                     'lon': row.geometry.x,
#                     'type': 'Condens Warmte',
#                     'name': row.get('Naam', 'N/A'),
#                     'gemeente': row.get('Plaats', 'N/A'),
#                     'color': '#32CD32',
#                     'TJ_MTWarmte': tj_mt_val,
#                     'power_display': f"{tj_mt_val:.2f} TJ" if tj_mt_val > 0 else "N/A"
#                 })
#             print(f"  ✓ Geocoded {geocoded_count} Condens Warmte locations from {len(unique_places)} unique places")
#     elif 'X' in cw_df.columns and 'Y' in cw_df.columns:
#         # Fallback to X,Y if available
#         cw_with_coords = cw_df.dropna(subset=['X', 'Y'])
#         if len(cw_with_coords) > 0:
#             gdf_cw = gpd.GeoDataFrame(cw_with_coords, geometry=gpd.points_from_xy(cw_with_coords['X'], cw_with_coords['Y']), crs='EPSG:28992').to_crs(epsg=4326)
#             for idx, row in gdf_cw.iterrows():
#                 tj_mt_val = row.get('TJ_MTWarmte', 0)
#                 try:
#                     tj_mt_val = float(tj_mt_val) if pd.notna(tj_mt_val) else 0.0
#                 except:
#                     tj_mt_val = 0.0
#                 all_warmte_sources.append({
#                     'lat': row.geometry.y,
#                     'lon': row.geometry.x,
#                     'type': 'Condens Warmte',
#                     'name': row.get('BronNaam', 'N/A'),
#                     'gemeente': row.get('Gemeente', 'N/A'),
#                     'color': '#32CD32',
#                     'TJ_MTWarmte': tj_mt_val,
#                     'power_display': f"{tj_mt_val:.2f} TJ" if tj_mt_val > 0 else "N/A"
#                 })
# --- END CONDENS WARMTE DATA ---

# Store geothermie GeoDataFrame for Defensie score calculation
geothermie_gdf = warmte_data.get("OVERVIEW_potential_recoverable_heat.nc", None)

print(f"  ✓ Collected {len(all_warmte_sources)} warmte sources for analytics")

# ============ RVB BUILDINGS ============
print("Adding RVB Buildings layer...")
rvb_group = folium.FeatureGroup(name='🏢 RVB Buildings', show=True)

# Store scores for Top 10 calculation
rvb_scores_for_top10 = []

# Create custom triangle icon
triangle_icon = folium.features.CustomIcon(
    icon_image='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBvbHlnb24gcG9pbnRzPSIxMCwyIDIsMTggMTgsMTgiIGZpbGw9IiMxYTU0OTAiIHN0cm9rZT0iIzAwMCIgc3Ryb2tlLXdpZHRoPSIxLjUiLz48L3N2Zz4=',
    icon_size=(20, 20),
    icon_anchor=(10, 10)
)
def make_triangle_icon(color_hex: str) -> folium.features.CustomIcon:
    svg = f"""
    <svg width="20" height="20" xmlns="http://www.w3.org/2000/svg">
      <polygon points="10,2 2,18 18,18" fill="{color_hex}" stroke="#000" stroke-width="1.5"/>
    </svg>
    """.strip()

    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return folium.features.CustomIcon(
        icon_image=f"data:image/svg+xml;base64,{b64}",
        icon_size=(20, 20),
        icon_anchor=(10, 10)
    )

for idx, row in rvb_points.iterrows():
    # Calculate distances to all warmte sources - only within 1km
    lat, lon = row.geometry.y, row.geometry.x
    nearby_sources = []

    for source in all_warmte_sources:
        km = haversine_km(lon, lat, source['lon'], source['lat'])
        if km <= 1.0:  # Within 1km only
            nearby_sources.append({**source, 'distance': km})

    nearby_sources.sort(key=lambda x: x['distance'])

    # Calculate warmte score for this location
    raw_score, score_breakdown = calculate_warmte_score(lat, lon, all_warmte_sources, geothermie_gdf, include_geothermie=False)
    normalized_score = normalize_score(raw_score)

    # Calculate oordeel verbruik with warmte score adjustment
    max_vermogen = row.get('Max vermogen verbruik', None)
    contractcapaciteit = row.get('Contractcapaciteit', None)

    # Initialize values for display
    totaal_verbruik = None
    besparing_warmte = 0
    nieuw_totaal_verbruik = None
    besparing_display = ""

    # Determine base oordeel
    if pd.isna(max_vermogen) or pd.isna(contractcapaciteit) or contractcapaciteit == 0:
        oordeel_base = "Onbekend"
        verbruik_ratio = None
        oordeel_color = "#808080"  # Gray
    else:
        totaal_verbruik = max_vermogen
        verbruik_ratio = (max_vermogen / contractcapaciteit) * 100
        if verbruik_ratio <= 80:
            oordeel_base = "Groen"
            oordeel_color = "#4CAF50"
        elif verbruik_ratio <= 100:
            oordeel_base = "Oranje"
            oordeel_color = "#FF9800"
        else:
            oordeel_base = "Rood"
            oordeel_color = "#F44336"

    # Adjust oordeel based on warmte score (higher score = better potential for heat savings)
    # Besparing warmte opwek = raw_score (in MW-equivalent)
    # Nieuw Totaal verbruik = Totaal verbruik - Besparing warmte opwek
    adjusted_oordeel = oordeel_base
    adjusted_color = oordeel_color
    adjusted_ratio = verbruik_ratio

    if raw_score > 0 and totaal_verbruik is not None:
        # Besparing is direct gerelateerd aan beschikbare warmte (raw_score is in MW-eq)
        # Cap de besparing op maximaal 50% van het totale verbruik
        besparing_warmte = min(raw_score, totaal_verbruik * 0.5)
        nieuw_totaal_verbruik = totaal_verbruik - besparing_warmte

        # Bereken nieuw oordeel op basis van nieuw verbruik
        if contractcapaciteit > 0:
            adjusted_ratio = (nieuw_totaal_verbruik / contractcapaciteit) * 100
            if adjusted_ratio <= 80:
                adjusted_oordeel = "Groen"
                adjusted_color = "#4CAF50"
            elif adjusted_ratio <= 100:
                adjusted_oordeel = "Oranje"
                adjusted_color = "#FF9800"
            else:
                adjusted_oordeel = "Rood"
                adjusted_color = "#F44336"

        besparing_display = f"-{besparing_warmte:.1f} MW ({(besparing_warmte/totaal_verbruik*100):.0f}%)" if totaal_verbruik > 0 else ""

    # Store data for Top 10 potentiële groei calculation
    # Potentiële groei = besparing potential (higher = more to gain from warmte)
    rvb_scores_for_top10.append({
        'code': row.get('BOUWWERKCO', 'N/A'),
        'naam': row.get('Objectnaam', row.get('BOUWWERKCO', 'N/A')),
        'besparing_mw': besparing_warmte,
        'raw_score': raw_score,
        'normalized_score': normalized_score,
        'totaal_verbruik': totaal_verbruik if totaal_verbruik else 0,
        'oordeel_base': oordeel_base,
        'oordeel_color_base': oordeel_color,
        'adjusted_oordeel': adjusted_oordeel,
        'adjusted_color': adjusted_color,
        'op_warmtenet': row.get('Op bestaand warmtenet', 'Nee'),
        'lat': row.geometry.y,
        'lon': row.geometry.x
    })

    # Build analytics HTML with power column
    sources_html = ""
    type_counts = {}
    total_power = {'MT Warmte': 0, 'Datacenter': 0, 'Condens Warmte': 0}

    for s in nearby_sources:
        type_counts[s['type']] = type_counts.get(s['type'], 0) + 1
        power_display = s.get('power_display', 'N/A')

        # Accumulate power by type
        if s['type'] == 'MT Warmte':
            total_power['MT Warmte'] += s.get('MWth', 0) or 0
        elif s['type'] == 'Datacenter':
            total_power['Datacenter'] += s.get('VERMOGEN_numeric', 0) or 0
        elif s['type'] == 'Condens Warmte':
            total_power['Condens Warmte'] += s.get('TJ_MTWarmte', 0) or 0

        sources_html += f"""
        <tr style="border-bottom: 1px solid #eee;">
            <td style="padding: 4px;"><span style="color: {s['color']};">●</span> {s['type']}</td>
            <td style="padding: 4px;">{s['name'][:18]}</td>
            <td style="padding: 4px;">{power_display}</td>
            <td style="padding: 4px;">{s['distance']*1000:.0f}m</td>
        </tr>
        """

    chart_html = ""
    for stype, count in type_counts.items():
        pct = (count / len(nearby_sources) * 100) if nearby_sources else 0
        chart_html += f'<div style="background: #e0e0e0; margin: 2px 0; border-radius: 3px;"><div style="background: linear-gradient(90deg, #1a5490, #42a5f5); width: {pct}%; padding: 2px 5px; color: white; font-size: 10px; border-radius: 3px;">{stype}: {count}</div></div>'

    # Normalize score and get interpretation
    normalized_score = normalize_score(raw_score)
    score_label, score_color, score_description = get_score_interpretation(normalized_score)
    score_gauge_html = create_score_gauge_html(normalized_score, score_label, score_color)

    # Build score breakdown HTML
    score_breakdown_html = f"""
    <div style="font-size: 10px; color: #666; margin-top: 8px; padding-top: 8px; border-top: 1px solid #eee;">
        <b>Componenten (ruwe waarden):</b><br>
        <table style="width: 100%; margin-top: 4px;">
            <tr><td>🌡️ MT Warmte:</td><td style="text-align: right;"><b>{score_breakdown['mt_warmte_mwth']:.1f} MW</b></td></tr>
            <tr><td>💻 Datacenter (>60°C):</td><td style="text-align: right;"><b>{score_breakdown['datacenter_vermogen']:.1f} MW</b></td></tr>
            <tr><td>❄️ Condens Warmte:</td><td style="text-align: right;"><b>{score_breakdown['condens_tj_mt']:.1f} TJ</b></td></tr>
        </table>
        <p style="margin: 6px 0 0 0; font-style: italic; color: #888;">Ruwe score: {raw_score:.2f} MW-eq</p>
    </div>
    """

    # Build the verbruik comparison HTML
    verbruik_html = ""
    if totaal_verbruik is not None:
        verbruik_html = f"""
        <div style="background: #f8f9fa; padding: 10px; border-radius: 6px; margin-top: 10px;">
            <table style="width: 100%; font-size: 12px;">
                <tr>
                    <td style="padding: 4px;"><b>Totaal Verbruik (huidig):</b></td>
                    <td style="text-align: right; color: {oordeel_color};"><b>{totaal_verbruik:.1f} MW</b></td>
                </tr>
                <tr style="color: #4CAF50;">
                    <td style="padding: 4px;"><b>Besparing warmte opwek:</b></td>
                    <td style="text-align: right;"><b>{besparing_display if besparing_display else '0 MW'}</b></td>
                </tr>
                <tr style="border-top: 2px solid #1a5490;">
                    <td style="padding: 4px;"><b>Nieuw Totaal Verbruik:</b></td>
                    <td style="text-align: right; color: {adjusted_color};"><b>{nieuw_totaal_verbruik:.1f} MW</b></td>
                </tr>
                <tr>
                    <td style="padding: 4px;">Nieuw Verbruik Ratio:</td>
                    <td style="text-align: right;"><b>{adjusted_ratio:.1f}%</b> van capaciteit</td>
                </tr>
            </table>
        </div>
        """ if nieuw_totaal_verbruik is not None else f"""
        <div style="background: #f8f9fa; padding: 10px; border-radius: 6px; margin-top: 10px;">
            <table style="width: 100%; font-size: 12px;">
                <tr>
                    <td style="padding: 4px;"><b>Totaal Verbruik:</b></td>
                    <td style="text-align: right;"><b>{totaal_verbruik:.1f} MW</b></td>
                </tr>
                <tr>
                    <td style="padding: 4px; color: #999;">Geen warmtebesparing mogelijk</td>
                    <td></td>
                </tr>
            </table>
        </div>
        """

    popup_html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; width: 520px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.3);">
        <h3 style="color: white; margin: 0 0 10px 0; font-weight: 600; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🏢 RVB Building
        </h3>
        <div style="background: white; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
            <table style="width: 100%; font-size: 12px;">
                <tr><td><b>Code:</b></td><td>{row.get('BOUWWERKCO', 'N/A')}</td></tr>
                <tr><td><b>EAN:</b></td><td>{row.get('EAN', 'N/A')}</td></tr>
                <tr><td><b>Bouwwerkfunctie:</b></td><td>{row.get('Bouwwerkfunctie', 'N/A')}</td></tr>
                <tr><td><b>Contractcapaciteit:</b></td><td>{contractcapaciteit if pd.notna(contractcapaciteit) else 'N/A'} MW</td></tr>
                <tr><td><b>WP aanwezig:</b></td><td>{row.get('WP aanwezig', 'N/A')}</td></tr>
                <tr><td><b>Op bestaand warmtenet:</b></td><td style="color: {'#4CAF50' if row.get('Op bestaand warmtenet') == 'Ja' else '#F44336'}; font-weight: bold;">{row.get('Op bestaand warmtenet', 'N/A')}</td></tr>
            </table>
        </div>

        <div style="background: white; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
            <h4 style="margin: 0 0 8px 0; color: #1a5490;">⚡ Oordeel Verbruik & Netcongestie Bijdrage</h4>
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                <div style="text-align: center;">
                    <div style="font-size: 10px; color: #666; margin-bottom: 4px;">HUIDIG</div>
                    <span style="background: {oordeel_color}; color: white; padding: 6px 14px; border-radius: 4px; font-weight: bold; display: inline-block;">{oordeel_base}</span>
                </div>
                <div style="font-size: 20px; color: #1a5490;">→</div>
                <div style="text-align: center;">
                    <div style="font-size: 10px; color: #666; margin-bottom: 4px;">MET WARMTE</div>
                    <span style="background: {adjusted_color}; color: white; padding: 6px 14px; border-radius: 4px; font-weight: bold; display: inline-block;">{adjusted_oordeel}</span>
                </div>
            </div>
            {verbruik_html}
        </div>

        <div style="background: white; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
            <h4 style="margin: 0 0 4px 0; color: #1a5490;">🔥 Warmte Besparing Potentieel</h4>
            <p style="margin: 0 0 8px 0; font-size: 11px; color: #666;">{score_description}</p>
            {score_gauge_html}
            {score_breakdown_html}
        </div>

        <div style="background: white; padding: 12px; border-radius: 8px;">
            <h4 style="margin: 0 0 8px 0; color: #1a5490;">📊 Nabije Warmtebronnen (&lt;1km)</h4>
            <div style="margin-bottom: 10px;">{chart_html}</div>
            <div style="max-height: 150px; overflow-y: auto;">
                <table style="width: 100%; font-size: 11px;">
                    <thead style="background: #f5f5f5; position: sticky; top: 0;">
                        <tr><th style="padding: 4px; text-align: left;">Type</th><th style="padding: 4px; text-align: left;">Naam</th><th style="padding: 4px; text-align: left;">Vermogen</th><th style="padding: 4px; text-align: left;">Afstand</th></tr>
                    </thead>
                    <tbody>{sources_html if sources_html else '<tr><td colspan="4" style="text-align: center; padding: 10px; color: #999;">Geen bronnen binnen 1km</td></tr>'}</tbody>
                </table>
            </div>
            <p style="margin: 10px 0 0 0; font-size: 10px; color: #666; text-align: center;">
                <b>Totaal: {len(nearby_sources)} bronnen binnen 1km</b>
            </p>
        </div>
    </div>
    """

    # Marker color reflects the adjusted oordeel (with warmte savings)
    marker_color = adjusted_color if besparing_warmte > 0 else oordeel_color

    folium.Marker(
        location=[row.geometry.y, row.geometry.x],
        popup=folium.Popup(popup_html, max_width=570),
        tooltip=f"🏢 RVB: {row.get('BOUWWERKCO', 'N/A')} | {oordeel_base}→{adjusted_oordeel} | Score: {normalized_score:.0f}/100",
        icon=make_triangle_icon(marker_color)
    ).add_to(rvb_group)



rvb_group.add_to(m)

# ============ CALCULATE TOP 10 POTENTIËLE GROEI ============
print("Calculating Top 10 potentiële groei...")

# Color improvement score: prioritize buildings with biggest color change (e.g., Rood → Groen)
def color_improvement_score(item):
    """Calculate score based on color change. Higher = better improvement."""
    color_rank = {'Rood': 0, 'Oranje': 1, 'Groen': 2, 'Onbekend': -1}
    base = color_rank.get(item['oordeel_base'], -1)
    adjusted = color_rank.get(item['adjusted_oordeel'], -1)
    if base == -1 or adjusted == -1:
        return 0  # Unknown stays low priority
    return adjusted - base  # Rood(0)→Groen(2) = 2, Rood(0)→Oranje(1) = 1, etc.

# Sort by color improvement first (highest first), then by besparing_mw as tiebreaker
top10_candidates = [r for r in rvb_scores_for_top10 if r['besparing_mw'] > 0]
top10_candidates.sort(key=lambda x: (color_improvement_score(x), x['besparing_mw']), reverse=True)
top10_groei = top10_candidates[:10]

# Build the Top 10 table HTML rows - need map variable name for flyTo
map_name = m.get_name()
top10_rows_html = ""
for i, item in enumerate(top10_groei, 1):
    base_color = item.get('oordeel_color_base', "#808080")
    adj_color = item.get('adjusted_color', "#808080")
    warmtenet_icon = "✓" if item['op_warmtenet'] == 'Ja' else "✗"
    warmtenet_color = "#4CAF50" if item['op_warmtenet'] == 'Ja' else "#999"
    lat, lon = item['lat'], item['lon']
    top10_rows_html += f"""
    <tr style="border-bottom: 1px solid rgba(255,255,255,0.1); cursor: pointer;" onclick="{map_name}.flyTo([{lat}, {lon}], 16);" title="Klik om naar locatie te gaan">
        <td style="padding: 6px 8px; color: rgba(255,255,255,0.9); font-weight: bold;">{i}</td>
        <td style="padding: 6px 8px; color: #64B5F6; max-width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; text-decoration: underline;" title="{item['naam']}">{item['code']}</td>
        <td style="padding: 6px 8px; color: #4CAF50; font-weight: bold;">{item['besparing_mw']:.1f}</td>
        <td style="padding: 6px 8px; color: rgba(255,255,255,0.7);">{item['totaal_verbruik']:.1f}</td>
        <td style="padding: 6px 8px;"><span style="color: {base_color};">●</span><span style="color: rgba(255,255,255,0.5);">→</span><span style="color: {adj_color};">●</span></td>
        <td style="padding: 6px 8px; color: {warmtenet_color};">{warmtenet_icon}</td>
    </tr>
    """

print(f"  Top 10 buildings with highest besparing potential identified")

# ============ DEFENSIE VKA - BOVENREGIONAAL ============
print("Adding Defensie VKA - Bovenregionaal layer...")
defensie_boven_group = folium.FeatureGroup(name='🛡️ Defensie VKA - Bovenregionaal', show=True)

for geojson_file in bovenregionaal_files:
    try:
        gdf_def = gpd.read_file(geojson_file)
        gdf_def_projected = gdf_def.to_crs(epsg=28992)
        gdf_def_projected["centroid"] = gdf_def_projected.geometry.centroid
        gdf_def_wgs84 = gdf_def_projected.to_crs(epsg=4326)
        gdf_def_wgs84["centroid_wgs84"] = gdf_def_projected["centroid"].to_crs(epsg=4326)

        filename = os.path.basename(geojson_file).replace('.geojson', '')

        for idx, row in gdf_def_wgs84.iterrows():
            lat, lon = row["centroid_wgs84"].y, row["centroid_wgs84"].x
            nearby_sources = []

            # Only include sources within 1km
            for source in all_warmte_sources:
                km = haversine_km(lon, lat, source['lon'], source['lat'])
                if km <= 1.0:  # Within 1km only
                    nearby_sources.append({**source, 'distance': km})

            nearby_sources.sort(key=lambda x: x['distance'])

            # Calculate warmte score WITH geothermie for Defensie
            raw_score, score_breakdown = calculate_warmte_score(lat, lon, all_warmte_sources, geothermie_gdf, include_geothermie=True)

            # Build analytics HTML with power column
            sources_html = ""
            type_counts = {}
            for s in nearby_sources:
                type_counts[s['type']] = type_counts.get(s['type'], 0) + 1
                power_display = s.get('power_display', 'N/A')
                sources_html += f'<tr style="border-bottom: 1px solid #eee;"><td style="padding: 4px;"><span style="color: {s["color"]};">●</span> {s["type"]}</td><td style="padding: 4px;">{s["name"][:18]}</td><td style="padding: 4px;">{power_display}</td><td style="padding: 4px;">{s["distance"]*1000:.0f}m</td></tr>'

            chart_html = ""
            for stype, count in type_counts.items():
                pct = (count / len(nearby_sources) * 100) if nearby_sources else 0
                chart_html += f'<div style="background: #e0e0e0; margin: 2px 0; border-radius: 3px;"><div style="background: linear-gradient(90deg, #1a5490, #42a5f5); width: {pct}%; padding: 2px 5px; color: white; font-size: 10px; border-radius: 3px;">{stype}: {count}</div></div>'

            # Normalize score and get interpretation
            normalized_score = normalize_score(raw_score)
            score_label, score_color, score_description = get_score_interpretation(normalized_score)
            score_gauge_html = create_score_gauge_html(normalized_score, score_label, score_color)

            # Build score breakdown HTML with geothermie
            score_breakdown_html = f"""
            <div style="font-size: 10px; color: #666; margin-top: 8px; padding-top: 8px; border-top: 1px solid #eee;">
                <b>Componenten (ruwe waarden):</b><br>
                <table style="width: 100%; margin-top: 4px;">
                    <tr><td>🌡️ MT Warmte:</td><td style="text-align: right;"><b>{score_breakdown['mt_warmte_mwth']:.1f} MW</b></td></tr>
                    <tr><td>💻 Datacenter (>60°C):</td><td style="text-align: right;"><b>{score_breakdown['datacenter_vermogen']:.1f} MW</b></td></tr>
                            <tr><td>❄️ Condens Warmte:</td><td style="text-align: right;"><b>{score_breakdown['condens_tj_mt']:.1f} TJ</b></td></tr>
                    <tr style="color: #FF8C00;"><td>🌋 Geothermie:</td><td style="text-align: right;"><b>{score_breakdown['geothermie_heat']:.2f}</b></td></tr>
                </table>
                <p style="margin: 6px 0 0 0; font-style: italic; color: #888;">Ruwe score: {raw_score:.2f} MW-eq</p>
            </div>
            """

            popup_html = f"""
            <div style="font-family: 'Segoe UI', Arial, sans-serif; width: 500px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.3);">
                <h3 style="color: white; margin: 0 0 10px 0; font-weight: 600; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                    🛡️ Defensie VKA - Bovenregionaal
                </h3>
                <div style="background: white; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
                    <table style="width: 100%; font-size: 12px;">
                        <tr><td><b>Naam:</b></td><td>{row.get('Naam', 'N/A')}</td></tr>
                        <tr><td><b>File:</b></td><td>{filename[:30]}</td></tr>
                    </table>
                </div>

                <div style="background: white; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
                    <h4 style="margin: 0 0 4px 0; color: #1a5490;">🔥 Warmte Besparing Potentieel (incl. Geothermie)</h4>
                    <p style="margin: 0 0 8px 0; font-size: 11px; color: #666;">{score_description}</p>
                    {score_gauge_html}
                    {score_breakdown_html}
                </div>

                <div style="background: white; padding: 12px; border-radius: 8px;">
                    <h4 style="margin: 0 0 8px 0; color: #1a5490;">📊 Nabije Warmtebronnen (&lt;1km)</h4>
                    <div style="margin-bottom: 10px;">{chart_html}</div>
                    <div style="max-height: 180px; overflow-y: auto;">
                        <table style="width: 100%; font-size: 11px;">
                            <thead style="background: #f5f5f5; position: sticky; top: 0;">
                                <tr><th style="padding: 4px; text-align: left;">Type</th><th style="padding: 4px; text-align: left;">Naam</th><th style="padding: 4px; text-align: left;">Vermogen</th><th style="padding: 4px; text-align: left;">Afstand</th></tr>
                            </thead>
                            <tbody>{sources_html if sources_html else '<tr><td colspan="4" style="text-align: center; padding: 10px; color: #999;">Geen bronnen binnen 1km</td></tr>'}</tbody>
                        </table>
                    </div>
                    <p style="margin: 10px 0 0 0; font-size: 10px; color: #666; text-align: center;">
                        <b>Totaal: {len(nearby_sources)} bronnen binnen 1km</b>
                    </p>
                </div>
            </div>
            """

            folium.Marker(
                location=[row["centroid_wgs84"].y, row["centroid_wgs84"].x],
                popup=folium.Popup(popup_html, max_width=550),
                tooltip=f"🛡️ Defensie: {row.get('Naam', filename)} | Warmte Score: {normalized_score:.0f}/100 ({score_label})",
                icon=triangle_icon
            ).add_to(defensie_boven_group)
    except Exception as e:
        print(f"  Skipped {os.path.basename(geojson_file)}: {str(e)[:50]}")

defensie_boven_group.add_to(m)

# ============ DEFENSIE VKA - LOCATIESPECIFIEK ============
print("Adding Defensie VKA - Locatiespecifiek layer...")
defensie_loc_group = folium.FeatureGroup(name='🛡️ Defensie VKA - Locatiespecifiek', show=True)

for geojson_file in locatiespecifiek_files:
    try:
        gdf_def = gpd.read_file(geojson_file)
        gdf_def_projected = gdf_def.to_crs(epsg=28992)
        gdf_def_projected["centroid"] = gdf_def_projected.geometry.centroid
        gdf_def_wgs84 = gdf_def_projected.to_crs(epsg=4326)
        gdf_def_wgs84["centroid_wgs84"] = gdf_def_projected["centroid"].to_crs(epsg=4326)

        filename = os.path.basename(geojson_file).replace('.geojson', '')

        for idx, row in gdf_def_wgs84.iterrows():
            lat, lon = row["centroid_wgs84"].y, row["centroid_wgs84"].x
            nearby_sources = []

            # Only include sources within 1km
            for source in all_warmte_sources:
                km = haversine_km(lon, lat, source['lon'], source['lat'])
                if km <= 1.0:  # Within 1km only
                    nearby_sources.append({**source, 'distance': km})

            nearby_sources.sort(key=lambda x: x['distance'])

            # Calculate warmte score WITH geothermie for Defensie
            raw_score, score_breakdown = calculate_warmte_score(lat, lon, all_warmte_sources, geothermie_gdf, include_geothermie=True)

            # Build analytics HTML with power column
            sources_html = ""
            type_counts = {}
            for s in nearby_sources:
                type_counts[s['type']] = type_counts.get(s['type'], 0) + 1
                power_display = s.get('power_display', 'N/A')
                sources_html += f'<tr style="border-bottom: 1px solid #eee;"><td style="padding: 4px;"><span style="color: {s["color"]};">●</span> {s["type"]}</td><td style="padding: 4px;">{s["name"][:18]}</td><td style="padding: 4px;">{power_display}</td><td style="padding: 4px;">{s["distance"]*1000:.0f}m</td></tr>'

            chart_html = ""
            for stype, count in type_counts.items():
                pct = (count / len(nearby_sources) * 100) if nearby_sources else 0
                chart_html += f'<div style="background: #e0e0e0; margin: 2px 0; border-radius: 3px;"><div style="background: linear-gradient(90deg, #1a5490, #42a5f5); width: {pct}%; padding: 2px 5px; color: white; font-size: 10px; border-radius: 3px;">{stype}: {count}</div></div>'

            # Normalize score and get interpretation
            normalized_score = normalize_score(raw_score)
            score_label, score_color, score_description = get_score_interpretation(normalized_score)
            score_gauge_html = create_score_gauge_html(normalized_score, score_label, score_color)

            # Build score breakdown HTML with geothermie
            score_breakdown_html = f"""
            <div style="font-size: 10px; color: #666; margin-top: 8px; padding-top: 8px; border-top: 1px solid #eee;">
                <b>Componenten (ruwe waarden):</b><br>
                <table style="width: 100%; margin-top: 4px;">
                    <tr><td>🌡️ MT Warmte:</td><td style="text-align: right;"><b>{score_breakdown['mt_warmte_mwth']:.1f} MW</b></td></tr>
                    <tr><td>💻 Datacenter (>60°C):</td><td style="text-align: right;"><b>{score_breakdown['datacenter_vermogen']:.1f} MW</b></td></tr>
                            <tr><td>❄️ Condens Warmte:</td><td style="text-align: right;"><b>{score_breakdown['condens_tj_mt']:.1f} TJ</b></td></tr>
                    <tr style="color: #FF8C00;"><td>🌋 Geothermie:</td><td style="text-align: right;"><b>{score_breakdown['geothermie_heat']:.2f}</b></td></tr>
                </table>
                <p style="margin: 6px 0 0 0; font-style: italic; color: #888;">Ruwe score: {raw_score:.2f} MW-eq</p>
            </div>
            """

            popup_html = f"""
            <div style="font-family: 'Segoe UI', Arial, sans-serif; width: 500px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.3);">
                <h3 style="color: white; margin: 0 0 10px 0; font-weight: 600; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                    🛡️ Defensie VKA - Locatiespecifiek
                </h3>
                <div style="background: white; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
                    <table style="width: 100%; font-size: 12px;">
                        <tr><td><b>Naam:</b></td><td>{row.get('Naam', 'N/A')}</td></tr>
                        <tr><td><b>File:</b></td><td>{filename[:30]}</td></tr>
                    </table>
                </div>

                <div style="background: white; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
                    <h4 style="margin: 0 0 4px 0; color: #1a5490;">🔥 Warmte Besparing Potentieel (incl. Geothermie)</h4>
                    <p style="margin: 0 0 8px 0; font-size: 11px; color: #666;">{score_description}</p>
                    {score_gauge_html}
                    {score_breakdown_html}
                </div>

                <div style="background: white; padding: 12px; border-radius: 8px;">
                    <h4 style="margin: 0 0 8px 0; color: #1a5490;">📊 Nabije Warmtebronnen (&lt;1km)</h4>
                    <div style="margin-bottom: 10px;">{chart_html}</div>
                    <div style="max-height: 180px; overflow-y: auto;">
                        <table style="width: 100%; font-size: 11px;">
                            <thead style="background: #f5f5f5; position: sticky; top: 0;">
                                <tr><th style="padding: 4px; text-align: left;">Type</th><th style="padding: 4px; text-align: left;">Naam</th><th style="padding: 4px; text-align: left;">Vermogen</th><th style="padding: 4px; text-align: left;">Afstand</th></tr>
                            </thead>
                            <tbody>{sources_html if sources_html else '<tr><td colspan="4" style="text-align: center; padding: 10px; color: #999;">Geen bronnen binnen 1km</td></tr>'}</tbody>
                        </table>
                    </div>
                    <p style="margin: 10px 0 0 0; font-size: 10px; color: #666; text-align: center;">
                        <b>Totaal: {len(nearby_sources)} bronnen binnen 1km</b>
                    </p>
                </div>
            </div>
            """

            folium.Marker(
                location=[row["centroid_wgs84"].y, row["centroid_wgs84"].x],
                popup=folium.Popup(popup_html, max_width=550),
                tooltip=f"🛡️ Defensie: {row.get('Naam', filename)} | Warmte Score: {normalized_score:.0f}/100 ({score_label})",
                icon=triangle_icon
            ).add_to(defensie_loc_group)
    except Exception as e:
        print(f"  Skipped {os.path.basename(geojson_file)}: {str(e)[:50]}")

defensie_loc_group.add_to(m)

# ============ WARMTE BRONNEN ============
print("Adding Warmte (heat) sources layer...")
warmte_group = folium.FeatureGroup(name='🌡️ Warmte Bronnen', show=False)

mt_warmte_file = 'Download-MT-Warmtebronnen startanalyse  (2024)-CSV.csv'
if mt_warmte_file in warmte_data:
    mt_df = warmte_data[mt_warmte_file]

    if 'X' in mt_df.columns and 'Y' in mt_df.columns:
        mt_with_coords = mt_df.dropna(subset=['X', 'Y'])

        if len(mt_with_coords) > 0:
            gdf_warmte = gpd.GeoDataFrame(
                mt_with_coords,
                geometry=gpd.points_from_xy(mt_with_coords['X'], mt_with_coords['Y']),
                crs='EPSG:28992'
            )
            gdf_warmte = gdf_warmte.to_crs(epsg=4326)

            for idx, row in gdf_warmte.iterrows():
                popup_html = f"""
                <div style="font-family: Arial; width: 280px;">
                    <h4 style="color: #1E90FF; margin-bottom: 10px; border-bottom: 2px solid #1E90FF;">
                        🌡️ Warmtebron (MT)
                    </h4>
                    <table style="width: 100%; font-size: 12px;">
                        <tr><td><b>Naam:</b></td><td>{row.get('BronNaam', 'N/A')}</td></tr>
                        <tr><td><b>Type:</b></td><td>{row.get('TypeBron', 'N/A')}</td></tr>
                        <tr><td><b>Gemeente:</b></td><td>{row.get('Gemeente', 'N/A')}</td></tr>
                    </table>
                </div>
                """

                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=6,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"Warmte: {row.get('BronNaam', 'N/A')}",
                    color='#000080',
                    fillColor='#1E90FF',
                    fillOpacity=0.7,
                    weight=2
                ).add_to(warmte_group)

            print(f"  ✓ Added {len(gdf_warmte)} MT warmte sources")

warmte_group.add_to(m)


# ============ DATACENTER WARMTE ============
print("Adding Datacenter Warmte layer...")
datacenter_warmte_group = folium.FeatureGroup(name='💻 Datacenter Warmte', show=False)

datacenter_file = 'Download-LT DataCentraWarmte-CSV.csv'
if datacenter_file in warmte_data:
    dc_df = warmte_data[datacenter_file]

    if 'X' in dc_df.columns and 'Y' in dc_df.columns:
        dc_with_coords = dc_df.dropna(subset=['X', 'Y'])

        if len(dc_with_coords) > 0:
            gdf_dc = gpd.GeoDataFrame(
                dc_with_coords,
                geometry=gpd.points_from_xy(dc_with_coords['X'], dc_with_coords['Y']),
                crs='EPSG:28992'
            )
            gdf_dc = gdf_dc.to_crs(epsg=4326)

            for idx, row in gdf_dc.iterrows():
                popup_html = f"""
                <div style="font-family: Arial; width: 280px;">
                    <h4 style="color: #9370DB; margin-bottom: 10px; border-bottom: 2px solid #9370DB;">
                        💻 Datacenter Warmte
                    </h4>
                    <table style="width: 100%; font-size: 12px;">
                        <tr><td><b>Naam:</b></td><td>{row.get('BronNaam', 'N/A')}</td></tr>
                        <tr><td><b>Gemeente:</b></td><td>{row.get('Gemeente', 'N/A')}</td></tr>
                    </table>
                </div>
                """

                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=7,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"Datacenter: {row.get('BronNaam', 'N/A')}",
                    color='#4B0082',
                    fillColor='#9370DB',
                    fillOpacity=0.7,
                    weight=2
                ).add_to(datacenter_warmte_group)

            print(f"  ✓ Added {len(gdf_dc)} datacenter warmte sources")

datacenter_warmte_group.add_to(m)

# ============ CONDENS WARMTE LAYER (COMMENTED OUT) ============
# To re-enable: uncomment the block below
# --- START CONDENS WARMTE LAYER ---
# print("Adding Condens Warmte layer...")
# condens_warmte_group = folium.FeatureGroup(name='❄️ Condens Warmte (Koelprocessen)', show=False)
#
# condens_file = 'Download-LT CondensWarmte uit Koelprocessen-CSV.csv'
# if condens_file in warmte_data:
#     cw_df = warmte_data[condens_file]
#
#     # This file has 'Plaats' instead of X,Y - use geocoding (cache already populated)
#     if 'Plaats' in cw_df.columns:
#         cw_with_plaats = cw_df.dropna(subset=['Plaats'])
#         geocoded_markers = []
#
#         for idx, row in cw_with_plaats.iterrows():
#             plaats = row['Plaats']
#             x, y = geocode_plaats(plaats, geocode_cache)
#             if x is not None and y is not None:
#                 geocoded_markers.append({
#                     'X': x,
#                     'Y': y,
#                     'Naam': row.get('Naam', 'N/A'),
#                     'Plaats': plaats,
#                     'TJ_CondWarmte': row.get('TJ_CondWarmte', 'N/A'),
#                     'SBINaam': row.get('SBINaam', 'N/A')
#                 })
#
#         if geocoded_markers:
#             geocoded_df = pd.DataFrame(geocoded_markers)
#             gdf_cw = gpd.GeoDataFrame(
#                 geocoded_df,
#                 geometry=gpd.points_from_xy(geocoded_df['X'], geocoded_df['Y']),
#                 crs='EPSG:28992'
#             )
#             gdf_cw = gdf_cw.to_crs(epsg=4326)
#
#             for idx, row in gdf_cw.iterrows():
#                 # Format TJ value for display
#                 tj_value = row.get('TJ_CondWarmte', 'N/A')
#                 if isinstance(tj_value, (int, float)):
#                     tj_display = f"{tj_value:.2f} TJ"
#                 else:
#                     tj_display = str(tj_value)
#
#                 popup_html = f"""
#                 <div style="font-family: Arial; width: 280px;">
#                     <h4 style="color: #32CD32; margin-bottom: 10px; border-bottom: 2px solid #32CD32;">
#                         ❄️ Condens Warmte (Koelprocessen)
#                     </h4>
#                     <table style="width: 100%; font-size: 12px;">
#                         <tr><td><b>Naam:</b></td><td>{row.get('Naam', 'N/A')}</td></tr>
#                         <tr><td><b>Sector:</b></td><td>{row.get('SBINaam', 'N/A')}</td></tr>
#                         <tr><td><b>Plaats:</b></td><td>{row.get('Plaats', 'N/A')}</td></tr>
#                         <tr><td><b>Condens Warmte:</b></td><td>{tj_display}</td></tr>
#                     </table>
#                 </div>
#                 """
#
#                 folium.CircleMarker(
#                     location=[row.geometry.y, row.geometry.x],
#                     radius=6,
#                     popup=folium.Popup(popup_html, max_width=300),
#                     tooltip=f"Condens: {row.get('Naam', 'N/A')} ({row.get('Plaats', '')})",
#                     color='#228B22',
#                     fillColor='#32CD32',
#                     fillOpacity=0.7,
#                     weight=2
#                 ).add_to(condens_warmte_group)
#
#             print(f"  ✓ Added {len(gdf_cw)} condens warmte sources (geocoded)")
#
#     elif 'X' in cw_df.columns and 'Y' in cw_df.columns:
#         # Fallback to X,Y if available
#         cw_with_coords = cw_df.dropna(subset=['X', 'Y'])
#
#         if len(cw_with_coords) > 0:
#             gdf_cw = gpd.GeoDataFrame(
#                 cw_with_coords,
#                 geometry=gpd.points_from_xy(cw_with_coords['X'], cw_with_coords['Y']),
#                 crs='EPSG:28992'
#             )
#             gdf_cw = gdf_cw.to_crs(epsg=4326)
#
#             for idx, row in gdf_cw.iterrows():
#                 popup_html = f"""
#                 <div style="font-family: Arial; width: 280px;">
#                     <h4 style="color: #32CD32; margin-bottom: 10px; border-bottom: 2px solid #32CD32;">
#                         ❄️ Condens Warmte
#                     </h4>
#                     <table style="width: 100%; font-size: 12px;">
#                         <tr><td><b>Naam:</b></td><td>{row.get('BronNaam', 'N/A')}</td></tr>
#                         <tr><td><b>Type:</b></td><td>{row.get('TypeBron', 'N/A')}</td></tr>
#                         <tr><td><b>Gemeente:</b></td><td>{row.get('Gemeente', 'N/A')}</td></tr>
#                     </table>
#                 </div>
#                 """
#
#                 folium.CircleMarker(
#                     location=[row.geometry.y, row.geometry.x],
#                     radius=6,
#                     popup=folium.Popup(popup_html, max_width=300),
#                     tooltip=f"Condens: {row.get('BronNaam', 'N/A')}",
#                     color='#228B22',
#                     fillColor='#32CD32',
#                     fillOpacity=0.7,
#                     weight=2
#                 ).add_to(condens_warmte_group)
#
#             print(f"  ✓ Added {len(gdf_cw)} condens warmte sources")
#
# condens_warmte_group.add_to(m)
# --- END CONDENS WARMTE LAYER ---

# ============ Geothermie LAYERS ============
print("Adding Geothermie layers...")
thermogis_group = folium.FeatureGroup(name='🌍 ThermoGIS Geothermie', show=False)

# --- NEW: NetCDF warmte grid as heatmap layer (only high values) ---
nc_key = "OVERVIEW_potential_recoverable_heat.nc"
if nc_key in warmte_data:
    gdf_heat = warmte_data[nc_key]

    # Build list [lat, lon, weight] for HeatMap
    heat_points = []
    heat_values = []
    for _, row in gdf_heat.iterrows():
        val = row["heat"]
        if pd.isna(val):
            continue
        # Ignore negative or very small values
        if val <= 0:
            continue
        heat_values.append(val)

    # Calculate 75th percentile threshold to only show high values
    if len(heat_values) > 0:
        threshold = np.percentile(heat_values, 75)  # Only show top 25%

        for _, row in gdf_heat.iterrows():
            val = row["heat"]
            if pd.isna(val) or val <= threshold:
                continue
            heat_points.append([row.geometry.y, row.geometry.x, float(val)])

        if heat_points:
            HeatMap(
                heat_points,
                name="🌡️ Potentieel herwinbare warmte (High Values Only)",
                radius=15,
                blur=20,
                max_zoom=12,
                gradient={0.4: 'yellow', 0.6: 'orange', 0.8: 'red', 1.0: 'darkred'}
            ).add_to(thermogis_group)

            print(f"  ✓ Added NetCDF warmte grid to map ({len(heat_points)} high-value cells, threshold: {threshold:.2f})")

thermogis_group.add_to(m)

#============= WarmteNet LAYER ============

# Add buurt_warmte_net areas to the map
Warmte_net_group = folium.FeatureGroup(name='🏘️ Warmte Net Areas', show=True)

for _, row in buurt_warmte_net.iterrows():
    # Create a GeoJSON feature with properties for the tooltip
    geojson_feature = {
        "type": "Feature",
        "geometry": row['geometry'].__geo_interface__,
        "properties": {
            "BU_CODE": row.get('BU_CODE', 'N/A'),
            "BU_NAAM": row.get('BU_NAAM', 'N/A'),
            "GM_NAAM": row.get('GM_NAAM', 'N/A'),
            "WONINGEN": row.get('WONINGEN', 'N/A')
        }
    }

    folium.GeoJson(
        geojson_feature,
        style_function=lambda x: {
            'fillColor': "#FF00C3",
            'color': "#FF227A",
            'weight': 1,
            'fillOpacity': 0.5
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['BU_CODE', 'BU_NAAM', 'GM_NAAM', 'WONINGEN'],
            aliases=['Buurt Code:', 'Buurt Naam:', 'Gemeente:', 'Woningen:'],
            localize=True
        )
    ).add_to(Warmte_net_group)

print(f"  Added {len(buurt_warmte_net)} warmte net polygons to map")
Warmte_net_group.add_to(m)


# ============ PLUGINS ============
minimap = MiniMap(toggle_display=True)
m.add_child(minimap)

Fullscreen(position='topright', title='Fullscreen', title_cancel='Exit').add_to(m)
MeasureControl(position='topleft', primary_length_unit='kilometers').add_to(m)
folium.LayerControl(position='topright', collapsed=False).add_to(m)

# ============ PROFESSIONAL TITLE ============
title_html = f'''
<div style="position: fixed; top: 15px; left: 50%; transform: translateX(-50%);
            width: 900px; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            border-radius: 16px; z-index: 9999;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            box-shadow: 0 12px 24px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.1);
            padding: 20px 30px; backdrop-filter: blur(10px);">
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div style="flex: 1;">
            <h2 style="margin: 0 0 8px 0; color: white; font-size: 28px; font-weight: 700; letter-spacing: -0.5px;">
                ⚡ Net Congestion Analytics Platform
            </h2>
            <p style="margin: 0; color: rgba(255,255,255,0.85); font-size: 13px; font-weight: 400;">
                Advanced Infrastructure & Energy Source Mapping • Netherlands
            </p>
        </div>
        <div style="text-align: right; padding-left: 20px;">
            <div style="background: rgba(255,255,255,0.15); padding: 8px 16px; border-radius: 8px; backdrop-filter: blur(5px);">
                <div style="color: white; font-size: 24px; font-weight: 700;">{len(rvb_points) + len(bovenregionaal_files) + len(locatiespecifiek_files)}</div>
                <div style="color: rgba(255,255,255,0.8); font-size: 11px; text-transform: uppercase; letter-spacing: 1px;">Locations</div>
            </div>
        </div>
    </div>
    <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.2); display: flex; justify-content: space-between; align-items: center;">
        <div style="color: rgba(255,255,255,0.7); font-size: 12px;">
            <span style="margin-right: 15px;">🏢 {len(rvb_points)} RVB</span>
            <span style="margin-right: 15px;">🛡️ {len(bovenregionaal_files) + len(locatiespecifiek_files)} Defensie</span>
            <span>🌡️ {len(all_warmte_sources)} Heat Sources</span>
        </div>
        <div style="color: rgba(255,255,255,0.6); font-size: 11px;">
            Click any location for detailed analytics →
        </div>
    </div>
</div>
'''
m.get_root().html.add_child(folium.Element(title_html))

# ============ PROFESSIONAL LEGEND ============
legend_html = '''
<div style="position: fixed; bottom: 30px; left: 30px; width: 340px;
            background: linear-gradient(135deg, rgba(30,60,114,0.95) 0%, rgba(42,82,152,0.95) 100%);
            border-radius: 12px; z-index: 9998;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            padding: 16px; box-shadow: 0 8px 16px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);">
    <h4 style="margin: 0 0 14px 0; color: white; font-size: 16px; font-weight: 600; letter-spacing: -0.3px;
                border-bottom: 1px solid rgba(255,255,255,0.2); padding-bottom: 10px;">
        📊 Data Legend
    </h4>

    <div style="background: rgba(255,255,255,0.08); padding: 10px; border-radius: 8px; margin-bottom: 12px;">
        <div style="color: rgba(255,255,255,0.9); font-size: 12px; font-weight: 600; margin-bottom: 8px;">Primary Locations</div>
        <div style="margin: 6px 0; display: flex; align-items: center;">
            <span style="display: inline-block; width: 0; height: 0;
                         border-left: 7px solid transparent; border-right: 7px solid transparent;
                         border-bottom: 12px solid #fff; margin-right: 12px; filter: drop-shadow(0 2px 3px rgba(0,0,0,0.3));"></span>
            <span style="color: rgba(255,255,255,0.95); font-size: 12px;">RVB Buildings & Defensie VKA</span>
        </div>
        <div style="color: rgba(255,255,255,0.6); font-size: 10px; margin-left: 26px;">Click for source analytics</div>
    </div>

    <div style="background: rgba(255,255,255,0.08); padding: 10px; border-radius: 8px; margin-bottom: 12px;">
        <div style="color: rgba(255,255,255,0.9); font-size: 12px; font-weight: 600; margin-bottom: 8px;">Heat Sources</div>
        <div style="margin: 5px 0; display: flex; align-items: center;">
            <span style="display: inline-block; width: 12px; height: 12px; background: #1E90FF;
                         margin-right: 10px; border-radius: 50%; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></span>
            <span style="color: rgba(255,255,255,0.85); font-size: 11px;">MT Warmte Bronnen</span>
        </div>
        <div style="margin: 5px 0; display: flex; align-items: center;">
            <span style="display: inline-block; width: 12px; height: 12px; background: #9370DB;
                         margin-right: 10px; border-radius: 50%; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></span>
            <span style="color: rgba(255,255,255,0.85); font-size: 11px;">Datacenter Warmte</span>
        </div>
        <div style="margin: 5px 0; display: flex; align-items: center;">
            <span style="display: inline-block; width: 12px; height: 12px; background: linear-gradient(135deg, yellow, orange, red);
                         margin-right: 10px; border-radius: 50%; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></span>
            <span style="color: rgba(255,255,255,0.85); font-size: 11px;">ThermoGIS Geothermie</span>
        </div>
    </div>

    <div style="background: rgba(255,255,255,0.08); padding: 10px; border-radius: 8px;">
        <div style="color: rgba(255,255,255,0.9); font-size: 12px; font-weight: 600; margin-bottom: 8px;">Areas</div>
        <div style="margin: 5px 0; display: flex; align-items: center;">
            <span style="display: inline-block; width: 12px; height: 12px; background: #FF00C3;
                         margin-right: 10px; border: 1px solid #FF227A; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></span>
            <span style="color: rgba(255,255,255,0.85); font-size: 11px;">Warmte Net Areas</span>
        </div>
    </div>

    <div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.15);
                color: rgba(255,255,255,0.5); font-size: 10px; text-align: center;">
        Data Sources: RVB • Defensie • TenNet • Warmteatlas • ThermoGIS
    </div>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# ============ TOP 10 POTENTIËLE GROEI PANEL ============
map_var_name = m.get_name()
top10_html = f'''
<style>
    #top10-panel tr:hover {{
        background: rgba(255,255,255,0.1) !important;
    }}
</style>
<div id="top10-panel" style="position: fixed; top: 10px; left: 10px; width: 400px;
            background: linear-gradient(135deg, rgba(30,60,114,0.95) 0%, rgba(42,82,152,0.95) 100%);
            border-radius: 12px; z-index: 9998;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            box-shadow: 0 8px 16px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.1);
            backdrop-filter: blur(10px); overflow: hidden;">
    <div id="top10-header" style="padding: 12px 16px; cursor: pointer; display: flex; justify-content: space-between; align-items: center;
                border-bottom: 1px solid rgba(255,255,255,0.2);"
         onclick="var content = document.getElementById('top10-content'); var arrow = document.getElementById('top10-arrow');
                  if(content.style.display === 'none') {{ content.style.display = 'block'; arrow.innerHTML = '▼'; }}
                  else {{ content.style.display = 'none'; arrow.innerHTML = '▶'; }}">
        <h4 style="margin: 0; color: white; font-size: 15px; font-weight: 600; letter-spacing: -0.3px;">
            🏆 Top 10 Potentiële Groei
        </h4>
        <span id="top10-arrow" style="color: white; font-size: 12px;">▼</span>
    </div>
    <div id="top10-content" style="padding: 12px 16px; max-height: 350px; overflow-y: auto;">
        <p style="color: rgba(255,255,255,0.6); font-size: 10px; margin: 0 0 10px 0;">
            Gebouwen met hoogste warmte-besparingspotentieel · Klik op rij om naar locatie te gaan
        </p>
        <table style="width: 100%; border-collapse: collapse; font-size: 11px;">
            <thead>
                <tr style="border-bottom: 2px solid rgba(255,255,255,0.2);">
                    <th style="padding: 6px 8px; text-align: left; color: rgba(255,255,255,0.7);">#</th>
                    <th style="padding: 6px 8px; text-align: left; color: rgba(255,255,255,0.7);">Code</th>
                    <th style="padding: 6px 8px; text-align: left; color: rgba(255,255,255,0.7);">Besp.</th>
                    <th style="padding: 6px 8px; text-align: left; color: rgba(255,255,255,0.7);">Verbr.</th>
                    <th style="padding: 6px 8px; text-align: left; color: rgba(255,255,255,0.7);">Oord.</th>
                    <th style="padding: 6px 8px; text-align: left; color: rgba(255,255,255,0.7);">WN</th>
                </tr>
            </thead>
            <tbody>
                {top10_rows_html}
            </tbody>
        </table>
        <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.15);
                    color: rgba(255,255,255,0.5); font-size: 9px;">
            <b>Legenda:</b> Besp. = Besparing (MW) | Verbr. = Verbruik (MW) | Oord. = Huidig→Met warmte | WN = Op Warmtenet
        </div>
    </div>
</div>
'''
m.get_root().html.add_child(folium.Element(top10_html))

# ============ SAVE AND OPEN ============
output_file = "comprehensive_energy_map.html"
m.save(output_file)
abs_path = os.path.abspath(output_file)

print("\n" + "=" * 80)
print("MAP CREATION COMPLETE!")
print("=" * 80)
print(f"✓ Output file: {output_file}")
print(f"✓ Full path: {abs_path}")
print(f"\n📊 DATA SUMMARY:")
print(f"  • RVB Buildings: {len(rvb_points)}")
print(f"  • Defensie Bovenregionaal: {len(bovenregionaal_files)} files")
print(f"  • Defensie Locatiespecifiek: {len(locatiespecifiek_files)} files")
print(f"  • TenNet datasets: {len(tennet_data)}")
print(f"  • Warmte datasets: {len(warmte_data)}")
print(f"\n🗺️ LAYERS ADDED:")
print(f"  ✓ Netherlands Boundary")
print(f"  ✓ RVB Buildings (energy-scaled)")
print(f"  ✓ Defensie VKA Bovenregionaal ({len(bovenregionaal_files)} areas)")
print(f"  ✓ Defensie VKA Locatiespecifiek ({len(locatiespecifiek_files)} areas)")
print(f"  ✓ TenNet Congestie (data loaded)")
print(f"  ✓ Warmte Bronnen (MT)")
print(f"  ✓ ThermoGIS Geothermie (WMS)")
print(f"  ✓ Warmte Net Areas")
print("=" * 80)

# Open in browser
webbrowser.open('file://' + abs_path)
print("\n✓ Map opened in browser!")
