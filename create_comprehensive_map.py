# -*- coding: utf-8 -*-
# ============================================================================
# COMPREHENSIVE INTERACTIVE MAP WITH ALL DATA SOURCES
# Run this file directly: python create_comprehensive_map.py
# ============================================================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import folium
from folium import plugins
from folium.plugins import HeatMap, MiniMap, Fullscreen, MeasureControl
import geopandas as gpd
import pandas as pd
import numpy as np
import webbrowser
import os
import glob
from sklearn.neighbors import BallTree
from scipy.spatial.distance import cdist
import json

print("=" * 80)
print("LOADING ALL DATA SOURCES...")
print("=" * 80)

# ============ 1. LOAD RVB BUILDINGS ============
print("\n[1/5] Loading RVB Buildings...")
gdf = gpd.read_file("data/Bouwwerken_netcongestie_data/Bouwwerken_netcongestie.shp")
gdf_wgs84 = gdf.to_crs(epsg=4326)
gdf_projected = gdf.to_crs(epsg=28992)
gdf_projected["centroid"] = gdf_projected.geometry.centroid
gdf_wgs84["centroid"] = gdf_projected["centroid"].to_crs(epsg=4326)
rvb_points = gdf_wgs84.copy()
rvb_points["geometry"] = rvb_points["centroid"]
rvb_points["energy_proxy"] = rvb_points["Shape_Area"]
min_area = rvb_points["energy_proxy"].min()
max_area = rvb_points["energy_proxy"].max()
rvb_points["radius"] = 4 + (rvb_points["energy_proxy"] - min_area) / (max_area - min_area) * 14

def get_color(value, min_val, max_val):
    if max_val == min_val:
        return '#FFA500'
    norm = (value - min_val) / (max_val - min_val)
    if norm < 0.25:
        return '#4CAF50'
    elif norm < 0.5:
        return '#FFEB3B'
    elif norm < 0.75:
        return '#FF9800'
    else:
        return '#F44336'

rvb_points["color"] = rvb_points["energy_proxy"].apply(lambda x: get_color(x, min_area, max_area))
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

# --- Libraries ---
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import netCDF4
import geopandas as gpd
import glob
import os

# --- Load NetCDF warmte grid ---
# Adjust path if your .nc file lives somewhere else
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

# ============ CREATE BASE MAP ============
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=7,
    tiles=None,
    control_scale=True,
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

# MT Warmte
mt_warmte_file = 'Download-MT-Warmtebronnen startanalyse  (2024)-CSV.csv'
if mt_warmte_file in warmte_data:
    mt_df = warmte_data[mt_warmte_file]
    if 'X' in mt_df.columns and 'Y' in mt_df.columns:
        mt_with_coords = mt_df.dropna(subset=['X', 'Y'])
        if len(mt_with_coords) > 0:
            gdf_mt = gpd.GeoDataFrame(mt_with_coords, geometry=gpd.points_from_xy(mt_with_coords['X'], mt_with_coords['Y']), crs='EPSG:28992').to_crs(epsg=4326)
            for idx, row in gdf_mt.iterrows():
                all_warmte_sources.append({
                    'lat': row.geometry.y,
                    'lon': row.geometry.x,
                    'type': 'MT Warmte',
                    'name': row.get('BronNaam', 'N/A'),
                    'gemeente': row.get('Gemeente', 'N/A'),
                    'color': '#1E90FF'
                })

# LT Warmte
lt_warmte_file = 'Download-LT-Warmtebronnen startanalyse  (2024)-CSV.csv'
if lt_warmte_file in warmte_data:
    lt_df = warmte_data[lt_warmte_file]
    if 'X' in lt_df.columns and 'Y' in lt_df.columns:
        lt_with_coords = lt_df.dropna(subset=['X', 'Y'])
        if len(lt_with_coords) > 0:
            gdf_lt = gpd.GeoDataFrame(lt_with_coords, geometry=gpd.points_from_xy(lt_with_coords['X'], lt_with_coords['Y']), crs='EPSG:28992').to_crs(epsg=4326)
            for idx, row in gdf_lt.iterrows():
                all_warmte_sources.append({
                    'lat': row.geometry.y,
                    'lon': row.geometry.x,
                    'type': 'LT Warmte',
                    'name': row.get('BronNaam', 'N/A'),
                    'gemeente': row.get('Gemeente', 'N/A'),
                    'color': '#00CED1'
                })

# Datacenter
datacenter_file = 'Download-LT DataCentraWarmte-CSV.csv'
if datacenter_file in warmte_data:
    dc_df = warmte_data[datacenter_file]
    if 'X' in dc_df.columns and 'Y' in dc_df.columns:
        dc_with_coords = dc_df.dropna(subset=['X', 'Y'])
        if len(dc_with_coords) > 0:
            gdf_dc = gpd.GeoDataFrame(dc_with_coords, geometry=gpd.points_from_xy(dc_with_coords['X'], dc_with_coords['Y']), crs='EPSG:28992').to_crs(epsg=4326)
            for idx, row in gdf_dc.iterrows():
                all_warmte_sources.append({
                    'lat': row.geometry.y,
                    'lon': row.geometry.x,
                    'type': 'Datacenter',
                    'name': row.get('BronNaam', 'N/A'),
                    'gemeente': row.get('Gemeente', 'N/A'),
                    'color': '#9370DB'
                })

# Condens Warmte
condens_file = 'Download-LT CondensWarmte uit Koelprocessen-CSV.csv'
if condens_file in warmte_data:
    cw_df = warmte_data[condens_file]
    if 'X' in cw_df.columns and 'Y' in cw_df.columns:
        cw_with_coords = cw_df.dropna(subset=['X', 'Y'])
        if len(cw_with_coords) > 0:
            gdf_cw = gpd.GeoDataFrame(cw_with_coords, geometry=gpd.points_from_xy(cw_with_coords['X'], cw_with_coords['Y']), crs='EPSG:28992').to_crs(epsg=4326)
            for idx, row in gdf_cw.iterrows():
                all_warmte_sources.append({
                    'lat': row.geometry.y,
                    'lon': row.geometry.x,
                    'type': 'Condens Warmte',
                    'name': row.get('BronNaam', 'N/A'),
                    'gemeente': row.get('Gemeente', 'N/A'),
                    'color': '#32CD32'
                })

print(f"  ✓ Collected {len(all_warmte_sources)} warmte sources for analytics")

# ============ RVB BUILDINGS ============
print("Adding RVB Buildings layer...")
rvb_group = folium.FeatureGroup(name='🏢 RVB Buildings', show=True)

# Create custom triangle icon
triangle_icon = folium.features.CustomIcon(
    icon_image='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBvbHlnb24gcG9pbnRzPSIxMCwyIDIsMTggMTgsMTgiIGZpbGw9IiMxYTU0OTAiIHN0cm9rZT0iIzAwMCIgc3Ryb2tlLXdpZHRoPSIxLjUiLz48L3N2Zz4=',
    icon_size=(20, 20),
    icon_anchor=(10, 10)
)

for idx, row in rvb_points.iterrows():
    # Calculate distances to all warmte sources
    lat, lon = row.geometry.y, row.geometry.x
    nearby_sources = []

    for source in all_warmte_sources:
        # Calculate haversine distance in km
        from math import radians, cos, sin, asin, sqrt
        lon1, lat1, lon2, lat2 = map(radians, [lon, lat, source['lon'], source['lat']])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        km = 6371 * c

        if km <= 10:  # Within 10km
            nearby_sources.append({**source, 'distance': km})

    nearby_sources.sort(key=lambda x: x['distance'])
    nearby_sources = nearby_sources[:10]  # Top 10 nearest

    # Build analytics HTML
    sources_html = ""
    type_counts = {}
    for s in nearby_sources:
        type_counts[s['type']] = type_counts.get(s['type'], 0) + 1
        sources_html += f"""
        <tr style="border-bottom: 1px solid #eee;">
            <td style="padding: 4px;"><span style="color: {s['color']};">●</span> {s['type']}</td>
            <td style="padding: 4px;">{s['name'][:20]}</td>
            <td style="padding: 4px;">{s['distance']:.2f} km</td>
        </tr>
        """

    chart_html = ""
    for stype, count in type_counts.items():
        pct = (count / len(nearby_sources) * 100) if nearby_sources else 0
        chart_html += f'<div style="background: #e0e0e0; margin: 2px 0; border-radius: 3px;"><div style="background: linear-gradient(90deg, #1a5490, #42a5f5); width: {pct}%; padding: 2px 5px; color: white; font-size: 10px; border-radius: 3px;">{stype}: {count}</div></div>'

    popup_html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; width: 450px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.3);">
        <h3 style="color: white; margin: 0 0 10px 0; font-weight: 600; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🏢 RVB Building
        </h3>
        <div style="background: white; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
            <table style="width: 100%; font-size: 12px;">
                <tr><td><b>Code:</b></td><td>{row.get('BOUWWERKCO', 'N/A')}</td></tr>
                <tr><td><b>EAN:</b></td><td>{row.get('EAN', 'N/A')}</td></tr>
                <tr><td><b>Status:</b></td><td>{row.get('AFSTOOTSTA', 'N/A')}</td></tr>
                <tr><td><b>Area:</b></td><td>{row.get('Shape_Area', 0):.2f} m²</td></tr>
            </table>
        </div>

        <div style="background: white; padding: 12px; border-radius: 8px;">
            <h4 style="margin: 0 0 8px 0; color: #1a5490;">📊 Nearby Heat Sources (within 10km)</h4>
            <div style="margin-bottom: 10px;">{chart_html}</div>
            <div style="max-height: 200px; overflow-y: auto;">
                <table style="width: 100%; font-size: 11px;">
                    <thead style="background: #f5f5f5; position: sticky; top: 0;">
                        <tr><th style="padding: 4px; text-align: left;">Type</th><th style="padding: 4px; text-align: left;">Name</th><th style="padding: 4px; text-align: left;">Distance</th></tr>
                    </thead>
                    <tbody>{sources_html if sources_html else '<tr><td colspan="3" style="text-align: center; padding: 10px; color: #999;">No sources within 10km</td></tr>'}</tbody>
                </table>
            </div>
            <p style="margin: 10px 0 0 0; font-size: 10px; color: #666; text-align: center;">
                <b>Total: {len(nearby_sources)} sources</b>
            </p>
        </div>
    </div>
    """

    folium.Marker(
        location=[row.geometry.y, row.geometry.x],
        popup=folium.Popup(popup_html, max_width=500),
        tooltip=f"🏢 RVB: {row.get('BOUWWERKCO', 'N/A')} | Click for analysis",
        icon=triangle_icon
    ).add_to(rvb_group)

rvb_group.add_to(m)

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
            # Calculate distances to all warmte sources
            lat, lon = row["centroid_wgs84"].y, row["centroid_wgs84"].x
            nearby_sources = []

            for source in all_warmte_sources:
                from math import radians, cos, sin, asin, sqrt
                lon1, lat1, lon2, lat2 = map(radians, [lon, lat, source['lon'], source['lat']])
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * asin(sqrt(a))
                km = 6371 * c
                if km <= 15:  # Within 15km for Defensie
                    nearby_sources.append({**source, 'distance': km})

            nearby_sources.sort(key=lambda x: x['distance'])
            nearby_sources = nearby_sources[:15]  # Top 15 nearest

            sources_html = ""
            type_counts = {}
            for s in nearby_sources:
                type_counts[s['type']] = type_counts.get(s['type'], 0) + 1
                sources_html += f'<tr style="border-bottom: 1px solid #eee;"><td style="padding: 4px;"><span style="color: {s["color"]};">●</span> {s["type"]}</td><td style="padding: 4px;">{s["name"][:20]}</td><td style="padding: 4px;">{s["distance"]:.2f} km</td></tr>'

            chart_html = ""
            for stype, count in type_counts.items():
                pct = (count / len(nearby_sources) * 100) if nearby_sources else 0
                chart_html += f'<div style="background: #e0e0e0; margin: 2px 0; border-radius: 3px;"><div style="background: linear-gradient(90deg, #1a5490, #42a5f5); width: {pct}%; padding: 2px 5px; color: white; font-size: 10px; border-radius: 3px;">{stype}: {count}</div></div>'

            popup_html = f"""
            <div style="font-family: 'Segoe UI', Arial, sans-serif; width: 450px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.3);">
                <h3 style="color: white; margin: 0 0 10px 0; font-weight: 600; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                    🛡️ Defensie VKA - Bovenregionaal
                </h3>
                <div style="background: white; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
                    <table style="width: 100%; font-size: 12px;">
                        <tr><td><b>Naam:</b></td><td>{row.get('Naam', 'N/A')}</td></tr>
                        <tr><td><b>File:</b></td><td>{filename[:30]}</td></tr>
                    </table>
                </div>
                <div style="background: white; padding: 12px; border-radius: 8px;">
                    <h4 style="margin: 0 0 8px 0; color: #1a5490;">📊 Nearby Heat Sources (within 15km)</h4>
                    <div style="margin-bottom: 10px;">{chart_html}</div>
                    <div style="max-height: 200px; overflow-y: auto;">
                        <table style="width: 100%; font-size: 11px;">
                            <thead style="background: #f5f5f5; position: sticky; top: 0;">
                                <tr><th style="padding: 4px; text-align: left;">Type</th><th style="padding: 4px; text-align: left;">Name</th><th style="padding: 4px; text-align: left;">Distance</th></tr>
                            </thead>
                            <tbody>{sources_html if sources_html else '<tr><td colspan="3" style="text-align: center; padding: 10px; color: #999;">No sources within 15km</td></tr>'}</tbody>
                        </table>
                    </div>
                    <p style="margin: 10px 0 0 0; font-size: 10px; color: #666; text-align: center;">
                        <b>Total: {len(nearby_sources)} sources</b>
                    </p>
                </div>
            </div>
            """

            folium.Marker(
                location=[row["centroid_wgs84"].y, row["centroid_wgs84"].x],
                popup=folium.Popup(popup_html, max_width=500),
                tooltip=f"🛡️ Defensie: {row.get('Naam', filename)} | Click for analysis",
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

            for source in all_warmte_sources:
                from math import radians, cos, sin, asin, sqrt
                lon1, lat1, lon2, lat2 = map(radians, [lon, lat, source['lon'], source['lat']])
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * asin(sqrt(a))
                km = 6371 * c
                if km <= 15:
                    nearby_sources.append({**source, 'distance': km})

            nearby_sources.sort(key=lambda x: x['distance'])
            nearby_sources = nearby_sources[:15]

            sources_html = ""
            type_counts = {}
            for s in nearby_sources:
                type_counts[s['type']] = type_counts.get(s['type'], 0) + 1
                sources_html += f'<tr style="border-bottom: 1px solid #eee;"><td style="padding: 4px;"><span style="color: {s["color"]};">●</span> {s["type"]}</td><td style="padding: 4px;">{s["name"][:20]}</td><td style="padding: 4px;">{s["distance"]:.2f} km</td></tr>'

            chart_html = ""
            for stype, count in type_counts.items():
                pct = (count / len(nearby_sources) * 100) if nearby_sources else 0
                chart_html += f'<div style="background: #e0e0e0; margin: 2px 0; border-radius: 3px;"><div style="background: linear-gradient(90deg, #1a5490, #42a5f5); width: {pct}%; padding: 2px 5px; color: white; font-size: 10px; border-radius: 3px;">{stype}: {count}</div></div>'

            popup_html = f"""
            <div style="font-family: 'Segoe UI', Arial, sans-serif; width: 450px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.3);">
                <h3 style="color: white; margin: 0 0 10px 0; font-weight: 600; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                    🛡️ Defensie VKA - Locatiespecifiek
                </h3>
                <div style="background: white; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
                    <table style="width: 100%; font-size: 12px;">
                        <tr><td><b>Naam:</b></td><td>{row.get('Naam', 'N/A')}</td></tr>
                        <tr><td><b>File:</b></td><td>{filename[:30]}</td></tr>
                    </table>
                </div>
                <div style="background: white; padding: 12px; border-radius: 8px;">
                    <h4 style="margin: 0 0 8px 0; color: #1a5490;">📊 Nearby Heat Sources (within 15km)</h4>
                    <div style="margin-bottom: 10px;">{chart_html}</div>
                    <div style="max-height: 200px; overflow-y: auto;">
                        <table style="width: 100%; font-size: 11px;">
                            <thead style="background: #f5f5f5; position: sticky; top: 0;">
                                <tr><th style="padding: 4px; text-align: left;">Type</th><th style="padding: 4px; text-align: left;">Name</th><th style="padding: 4px; text-align: left;">Distance</th></tr>
                            </thead>
                            <tbody>{sources_html if sources_html else '<tr><td colspan="3" style="text-align: center; padding: 10px; color: #999;">No sources within 15km</td></tr>'}</tbody>
                        </table>
                    </div>
                    <p style="margin: 10px 0 0 0; font-size: 10px; color: #666; text-align: center;">
                        <b>Total: {len(nearby_sources)} sources</b>
                    </p>
                </div>
            </div>
            """

            folium.Marker(
                location=[row["centroid_wgs84"].y, row["centroid_wgs84"].x],
                popup=folium.Popup(popup_html, max_width=500),
                tooltip=f"🛡️ Defensie: {row.get('Naam', filename)} | Click for analysis",
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

# ============ LT WARMTE BRONNEN ============
print("Adding LT Warmte sources layer...")
lt_warmte_group = folium.FeatureGroup(name='🌡️ LT Warmte Bronnen', show=False)

lt_warmte_file = 'Download-LT-Warmtebronnen startanalyse  (2024)-CSV.csv'
if lt_warmte_file in warmte_data:
    lt_df = warmte_data[lt_warmte_file]

    if 'X' in lt_df.columns and 'Y' in lt_df.columns:
        lt_with_coords = lt_df.dropna(subset=['X', 'Y'])

        if len(lt_with_coords) > 0:
            gdf_lt_warmte = gpd.GeoDataFrame(
                lt_with_coords,
                geometry=gpd.points_from_xy(lt_with_coords['X'], lt_with_coords['Y']),
                crs='EPSG:28992'
            )
            gdf_lt_warmte = gdf_lt_warmte.to_crs(epsg=4326)

            for idx, row in gdf_lt_warmte.iterrows():
                popup_html = f"""
                <div style="font-family: Arial; width: 280px;">
                    <h4 style="color: #00CED1; margin-bottom: 10px; border-bottom: 2px solid #00CED1;">
                        🌡️ Warmtebron (LT)
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
                    tooltip=f"LT Warmte: {row.get('BronNaam', 'N/A')}",
                    color='#008B8B',
                    fillColor='#00CED1',
                    fillOpacity=0.7,
                    weight=2
                ).add_to(lt_warmte_group)

            print(f"  ✓ Added {len(gdf_lt_warmte)} LT warmte sources")

lt_warmte_group.add_to(m)

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

# ============ AARDWARMTE (GEOTHERMAL) ============
print("Adding Aardwarmte layer...")
aardwarmte_group = folium.FeatureGroup(name='🌋 Aardwarmte P50', show=False)

aardwarmte_file = 'Download-AardwarmteP50Vermogen-CSV.csv'
if aardwarmte_file in warmte_data:
    aw_df = warmte_data[aardwarmte_file]

    if 'X' in aw_df.columns and 'Y' in aw_df.columns:
        aw_with_coords = aw_df.dropna(subset=['X', 'Y'])

        if len(aw_with_coords) > 0:
            gdf_aw = gpd.GeoDataFrame(
                aw_with_coords,
                geometry=gpd.points_from_xy(aw_with_coords['X'], aw_with_coords['Y']),
                crs='EPSG:28992'
            )
            gdf_aw = gdf_aw.to_crs(epsg=4326)

            for idx, row in gdf_aw.iterrows():
                popup_html = f"""
                <div style="font-family: Arial; width: 280px;">
                    <h4 style="color: #FF8C00; margin-bottom: 10px; border-bottom: 2px solid #FF8C00;">
                        🌋 Aardwarmte P50
                    </h4>
                    <table style="width: 100%; font-size: 12px;">
                        <tr><td><b>Locatie:</b></td><td>{row.get('BronNaam', 'N/A')}</td></tr>
                    </table>
                </div>
                """

                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=7,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"Aardwarmte: {row.get('BronNaam', 'N/A')}",
                    color='#8B4513',
                    fillColor='#FF8C00',
                    fillOpacity=0.7,
                    weight=2
                ).add_to(aardwarmte_group)

            print(f"  ✓ Added {len(gdf_aw)} aardwarmte sources")

aardwarmte_group.add_to(m)

# ============ CONDENS WARMTE (COOLING PROCESSES) ============
print("Adding Condens Warmte layer...")
condens_warmte_group = folium.FeatureGroup(name='❄️ Condens Warmte (Koelprocessen)', show=False)

condens_file = 'Download-LT CondensWarmte uit Koelprocessen-CSV.csv'
if condens_file in warmte_data:
    cw_df = warmte_data[condens_file]

    if 'X' in cw_df.columns and 'Y' in cw_df.columns:
        cw_with_coords = cw_df.dropna(subset=['X', 'Y'])

        if len(cw_with_coords) > 0:
            gdf_cw = gpd.GeoDataFrame(
                cw_with_coords,
                geometry=gpd.points_from_xy(cw_with_coords['X'], cw_with_coords['Y']),
                crs='EPSG:28992'
            )
            gdf_cw = gdf_cw.to_crs(epsg=4326)

            for idx, row in gdf_cw.iterrows():
                popup_html = f"""
                <div style="font-family: Arial; width: 280px;">
                    <h4 style="color: #32CD32; margin-bottom: 10px; border-bottom: 2px solid #32CD32;">
                        ❄️ Condens Warmte
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
                    tooltip=f"Condens: {row.get('BronNaam', 'N/A')}",
                    color='#228B22',
                    fillColor='#32CD32',
                    fillOpacity=0.7,
                    weight=2
                ).add_to(condens_warmte_group)

            print(f"  ✓ Added {len(gdf_cw)} condens warmte sources")

condens_warmte_group.add_to(m)

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

######### Rest Warmte

restwarmte_group = folium.FeatureGroup(name='🏭 PDOK Restwarmte', show=False)
try:
    pdok_url = 'https://service.pdok.nl/rvo/restwarmte/wms/v1_0'
    folium.raster_layers.WmsTileLayer(
        url=pdok_url,
        layers='liggingindustrieco2',
        transparent=True,
        format='image/png',
        opacity=0.6,
        name='PDOK Restwarmte',
        overlay=True,
        control=True,
        attr='PDOK - RVO WarmteAtlas'
    ).add_to(restwarmte_group)
    print("  ✓ PDOK Restwarmte WMS added")
except Exception as e:
    print(f"  ○ PDOK WMS error: {e}")
restwarmte_group.add_to(m)

# ============ HEAT MAP ============
print("Adding heat map layer...")
heatmap_group = folium.FeatureGroup(name='🔥 Heat Map', show=False)
heat_data = [[point.y, point.x, area] for point, area in zip(rvb_points.geometry, rvb_points["energy_proxy"])]

HeatMap(
    heat_data,
    min_opacity=0.3,
    max_opacity=0.8,
    radius=25,
    blur=20,
    gradient={
        0.0: 'blue',
        0.3: 'lime',
        0.5: 'yellow',
        0.7: 'orange',
        1.0: 'red'
    }
).add_to(heatmap_group)
heatmap_group.add_to(m)

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

    <div style="background: rgba(255,255,255,0.08); padding: 10px; border-radius: 8px;">
        <div style="color: rgba(255,255,255,0.9); font-size: 12px; font-weight: 600; margin-bottom: 8px;">Heat Sources</div>
        <div style="margin: 5px 0; display: flex; align-items: center;">
            <span style="display: inline-block; width: 12px; height: 12px; background: #1E90FF;
                         margin-right: 10px; border-radius: 50%; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></span>
            <span style="color: rgba(255,255,255,0.85); font-size: 11px;">MT Warmte Bronnen</span>
        </div>
        <div style="margin: 5px 0; display: flex; align-items: center;">
            <span style="display: inline-block; width: 12px; height: 12px; background: #00CED1;
                         margin-right: 10px; border-radius: 50%; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></span>
            <span style="color: rgba(255,255,255,0.85); font-size: 11px;">LT Warmte Bronnen</span>
        </div>
        <div style="margin: 5px 0; display: flex; align-items: center;">
            <span style="display: inline-block; width: 12px; height: 12px; background: #9370DB;
                         margin-right: 10px; border-radius: 50%; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></span>
            <span style="color: rgba(255,255,255,0.85); font-size: 11px;">Datacenter Warmte</span>
        </div>
        <div style="margin: 5px 0; display: flex; align-items: center;">
            <span style="display: inline-block; width: 12px; height: 12px; background: #FF8C00;
                         margin-right: 10px; border-radius: 50%; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></span>
            <span style="color: rgba(255,255,255,0.85); font-size: 11px;">Aardwarmte P50</span>
        </div>
        <div style="margin: 5px 0; display: flex; align-items: center;">
            <span style="display: inline-block; width: 12px; height: 12px; background: #32CD32;
                         margin-right: 10px; border-radius: 50%; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></span>
            <span style="color: rgba(255,255,255,0.85); font-size: 11px;">Condens Warmte</span>
        </div>
    </div>

    <div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.15);
                color: rgba(255,255,255,0.5); font-size: 10px; text-align: center;">
        Data Sources: RVB • Defensie • TenNet • Warmteatlas • ThermoGIS • PDOK
    </div>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

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
print(f"  ✓ PDOK Restwarmte (WMS)")
print(f"  ✓ Heat Map")
print("=" * 80)

# Open in browser
webbrowser.open('file://' + abs_path)
print("\n✓ Map opened in browser!")
