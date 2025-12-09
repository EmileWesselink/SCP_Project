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

# ============ RVB BUILDINGS ============
print("Adding RVB Buildings layer...")
rvb_group = folium.FeatureGroup(name='🏢 RVB Buildings', show=True)

for idx, row in rvb_points.iterrows():
    popup_html = f"""
    <div style="font-family: Arial; width: 280px;">
        <h4 style="color: #1a5490; margin-bottom: 10px; border-bottom: 2px solid #1a5490;">
            RVB Bouwwerk
        </h4>
        <table style="width: 100%; font-size: 12px;">
            <tr><td><b>Code:</b></td><td>{row.get('BOUWWERKCO', 'N/A')}</td></tr>
            <tr><td><b>EAN:</b></td><td>{row.get('EAN', 'N/A')}</td></tr>
            <tr><td><b>Status:</b></td><td>{row.get('AFSTOOTSTA', 'N/A')}</td></tr>
            <tr><td><b>Oppervlakte:</b></td><td>{row.get('Shape_Area', 0):.2f} m²</td></tr>
        </table>
    </div>
    """

    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=row["radius"],
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=f"RVB: {row.get('BOUWWERKCO', 'N/A')}",
        color='#000000',
        fillColor=row["color"],
        fillOpacity=0.7,
        weight=1.5
    ).add_to(rvb_group)

rvb_group.add_to(m)

# ============ DEFENSIE VKA - BOVENREGIONAAL ============
print("Adding Defensie VKA - Bovenregionaal layer...")
defensie_boven_group = folium.FeatureGroup(name='🛡️ Defensie VKA - Bovenregionaal', show=False)

for geojson_file in bovenregionaal_files:
    try:
        gdf_def = gpd.read_file(geojson_file)
        gdf_def = gdf_def.to_crs(epsg=4326)

        filename = os.path.basename(geojson_file).replace('.geojson', '')

        folium.GeoJson(
            gdf_def,
            name=filename,
            style_function=lambda x: {
                'fillColor': '#8B0000',
                'color': '#5d0000',
                'weight': 2,
                'fillOpacity': 0.4
            },
            tooltip=folium.GeoJsonTooltip(fields=['Naam'] if 'Naam' in gdf_def.columns else [], aliases=['Naam:'])
        ).add_to(defensie_boven_group)
    except Exception as e:
        print(f"  Skipped {os.path.basename(geojson_file)}: {str(e)[:50]}")

defensie_boven_group.add_to(m)

# ============ DEFENSIE VKA - LOCATIESPECIFIEK ============
print("Adding Defensie VKA - Locatiespecifiek layer...")
defensie_loc_group = folium.FeatureGroup(name='🛡️ Defensie VKA - Locatiespecifiek', show=False)

for geojson_file in locatiespecifiek_files:
    try:
        gdf_def = gpd.read_file(geojson_file)
        gdf_def = gdf_def.to_crs(epsg=4326)

        filename = os.path.basename(geojson_file).replace('.geojson', '')

        folium.GeoJson(
            gdf_def,
            name=filename,
            style_function=lambda x: {
                'fillColor': '#DC143C',
                'color': '#8B0000',
                'weight': 2,
                'fillOpacity': 0.5
            },
            tooltip=folium.GeoJsonTooltip(fields=['Naam'] if 'Naam' in gdf_def.columns else [], aliases=['Naam:'])
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
                    <h4 style="color: #FF6347; margin-bottom: 10px; border-bottom: 2px solid #FF6347;">
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
                    color='#8B0000',
                    fillColor='#FF6347',
                    fillOpacity=0.7,
                    weight=2
                ).add_to(warmte_group)

            print(f"  ✓ Added {len(gdf_warmte)} MT warmte sources")

warmte_group.add_to(m)

# ============ Geothermie LAYERS ============
print("Adding Geothermie layers...")
thermogis_group = folium.FeatureGroup(name='🌍 ThermoGIS Geothermie', show=False)

# --- NEW: NetCDF warmte grid as heatmap layer ---
nc_key = "OVERVIEW_potential_recoverable_heat.nc"
if nc_key in warmte_data:
    gdf_heat = warmte_data[nc_key]

    # Build list [lat, lon, weight] for HeatMap
    heat_points = []
    for _, row in gdf_heat.iterrows():
        val = row["heat"]
        if pd.isna(val):
            continue
        # Optional: ignore negative or very small values
        if val <= 0:
            continue
        heat_points.append([row.geometry.y, row.geometry.x, float(val)])

    if heat_points:
        HeatMap(
            heat_points,
            name="🌡️ Potentieel herwinbare warmte (grid)",
            radius=10,
            blur=15,
            max_zoom=12
        ).add_to(thermogis_group)

        print(f"  ✓ Added NetCDF warmte grid to map ({len(heat_points)} cells)")

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

# ============ TITLE ============
title_html = f'''
<div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
            width: 800px; height: 110px; background-color: white;
            border: 3px solid #1a5490; border-radius: 10px; z-index: 9999;
            font-family: Arial; box-shadow: 0 4px 6px rgba(0,0,0,0.1); padding: 10px;">
    <h3 style="margin: 5px 0; text-align: center; color: #1a5490; font-size: 24px;">
        🔌 COMPREHENSIVE ENERGY & INFRASTRUCTURE MAP
    </h3>
    <p style="margin: 5px 0; text-align: center; color: #546e7a; font-size: 13px; font-style: italic;">
        RVB Buildings • Defensie VKA • TenNet • Warmte • Geothermie • Restwarmte
    </p>
    <p style="margin: 5px 0; text-align: center; color: #37474f; font-size: 12px;">
        <b>{len(rvb_points)} RVB Buildings</b> •
        <b>{len(bovenregionaal_files)} Bovenreg. VKA</b> •
        <b>{len(locatiespecifiek_files)} Locatie VKA</b>
    </p>
    <p style="margin: 5px 0; text-align: center; color: #666; font-size: 11px;">
        Use layer control (top right) to toggle layers
    </p>
</div>
'''
m.get_root().html.add_child(folium.Element(title_html))

# ============ LEGEND ============
legend_html = '''
<div style="position: fixed; bottom: 50px; left: 50px; width: 300px;
            background-color: white; border: 2px solid #1a5490; border-radius: 8px;
            z-index: 9998; font-family: Arial; padding: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    <h4 style="margin: 0 0 10px 0; color: #1a5490; border-bottom: 2px solid #1a5490; padding-bottom: 5px;">
        Legenda
    </h4>
    <div style="margin: 5px 0;">
        <span style="display: inline-block; width: 15px; height: 15px; background: #4CAF50;
                     border: 1px solid #000; margin-right: 8px; border-radius: 50%;"></span>
        <span style="font-size: 11px;">RVB - Low Energy</span>
    </div>
    <div style="margin: 5px 0;">
        <span style="display: inline-block; width: 15px; height: 15px; background: #F44336;
                     border: 1px solid #000; margin-right: 8px; border-radius: 50%;"></span>
        <span style="font-size: 11px;">RVB - High Energy</span>
    </div>
    <div style="margin: 5px 0;">
        <span style="display: inline-block; width: 15px; height: 15px; background: #8B0000;
                     border: 1px solid #000; margin-right: 8px;"></span>
        <span style="font-size: 11px;">Defensie VKA Bovenregionaal</span>
    </div>
    <div style="margin: 5px 0;">
        <span style="display: inline-block; width: 15px; height: 15px; background: #DC143C;
                     border: 1px solid #000; margin-right: 8px;"></span>
        <span style="font-size: 11px;">Defensie VKA Locatiespecifiek</span>
    </div>
    <div style="margin: 5px 0;">
        <span style="display: inline-block; width: 15px; height: 15px; background: #FF6347;
                     border: 1px solid #000; margin-right: 8px; border-radius: 50%;"></span>
        <span style="font-size: 11px;">Warmte Bronnen (MT)</span>
    </div>
    <hr style="margin: 10px 0; border: none; border-top: 1px solid #ccc;">
    <p style="margin: 5px 0; font-size: 9px; color: #78909c; font-style: italic;">
        Data: RVB, Defensie, TenNet, Warmteatlas, ThermoGIS, PDOK
    </p>
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
