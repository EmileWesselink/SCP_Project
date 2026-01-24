# SCP Project - Netcongestie Analyse Platform

Een interactief kaartplatform voor het analyseren van warmtenetpotentieel en energieverbruik van Nederlandse overheidspanden (RVB), ter ondersteuning van beslissingen om netcongestie te verminderen door gebruik te maken van lokale warmtebronnen.

## Snel Starten

```bash
# 1. Clone de repository
git clone https://github.com/your-username/SCP_Project.git
cd SCP_Project

# 2. Maak een virtuele omgeving
python -m venv .venv
source .venv/bin/activate  # Op Windows: .venv\Scripts\activate

# 3. Installeer afhankelijkheden
pip install -r requirements.txt

# 4. Voer het script uit
python create_comprehensive_map.py
```

Het script genereert `comprehensive_energy_map.html` en opent dit in uw browser.

---

## Projectstructuur

```
SCP_Project/
├── create_comprehensive_map.py   # Hoofdscript - genereert de interactieve kaart
├── requirements.txt              # Python afhankelijkheden
├── README.md                     # Dit bestand (technische documentatie)
├── MANUAL.md                     # Gebruikershandleiding (niet-technisch, Nederlands)
├── comprehensive_energy_map.html # Gegenereerde output (na uitvoeren script)
└── data/                         # Data directory
    ├── Bouwwerken_netcongestie_data/   # RVB gebouwen shapefile
    ├── Country_data/                    # Nederland landsgrens
    ├── defensie_data/                   # Defensie VKA GeoJSON bestanden
    ├── Netherlands_shapefile/           # NL administratieve grens
    ├── tennet_data/                     # TenNet netcongestie data
    ├── TUD_data/                        # Energieverbruik data (TU Delft)
    ├── warmte_data/                     # Warmtebron datasets (CSV, NetCDF)
    └── Warmte_net_data/                 # Warmtenet data (Excel, Shapefile)
```

---

## Functionaliteiten

### Kaartlagen
- **RVB Gebouwen** - Overheidspanden met energieverbruik analyse
- **Defensie VKA** - Militaire locaties (Bovenregionaal & Locatiespecifiek)
- **Warmtenet Gebieden** - Bestaande warmtenet dekkingspolygonen
- **Warmtebronnen** - MT Warmte, Datacenters, Industriële restwarmte, Geothermie

### Analyses per Gebouw
- **Warmte Score (0-100)** - Warmtebesparingspotentieel gebaseerd op nabijgelegen bronnen (<1km)
- **Oordeel Verbruik** - Huidige vs. potentiële verbruiksbeoordeling (Groen/Oranje/Rood)
- **Op bestaand warmtenet** - Of het gebouw binnen een bestaand warmtenet ligt (Ja/Nee)
- **Besparingspotentieel** - Geschatte MW besparing door lokale warmtebenutting

### Dashboard Panelen
- **Top 10 Potentiële Groei** - Gebouwen met hoogste warmtebesparingspotentieel
- **Data Legenda** - Visuele gids voor kaartmarkeringen en kleuren

---

## Databronnen

| Dataset | Bron | Formaat | Locatie |
|---------|------|---------|---------|
| RVB Gebouwen | Rijksvastgoedbedrijf | Shapefile | `data/Bouwwerken_netcongestie_data/` |
| Energieverbruik | TU Delft (TUD) | Excel | `data/TUD_data/` |
| MT Warmte | PBL Startanalyse 2024 | CSV | `data/warmte_data/` |
| Datacenter Warmte | RVO | CSV | `data/warmte_data/` |
| PDOK Restwarmte | PDOK WFS Service | GeoJSON (live) | Opgehaald via API |
| Geothermie | ThermoGIS | NetCDF | `data/warmte_data/` |
| Warmtenetten | CBS Buurtkaart 2020 | Shapefile + Excel | `data/Warmte_net_data/` |
| Defensie VKA | Defensie | GeoJSON | `data/defensie_data/` |
| Nederland Grens | GADM | Shapefile | `data/Country_data/` |

---

## Vereisten

- Python 3.8+
- Afhankelijkheden (zie `requirements.txt`):

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

Installeren met:
```bash
pip install -r requirements.txt
```

---

## Configuratie

Het script gebruikt relatieve paden vanaf de projectroot. Benodigde databestanden:

```
data/Bouwwerken_netcongestie_data/Bouwwerken_netcongestie.shp
data/TUD_data/TUD_Basislijst_Bekende_aansluitingen_(sept25).xlsx
data/Warmte_net_data/Download-WarmteNetten-XLS.xlsx
data/Warmte_net_data/Buurtkaart_2020_v3/*.shp
data/warmte_data/*.csv
data/warmte_data/OVERVIEW_potential_recoverable_heat.nc
```

---

## Nieuwe Datalagen Toevoegen

Deze sectie legt uit hoe u nieuwe databronnen aan de kaart kunt toevoegen.

### Overzicht

De kaart gebruikt [Folium](https://python-visualization.github.io/folium/) voor visualisatie. Elke laag is een `FeatureGroup` die aan/uit gezet kan worden via het lagenmenu.

### Stap 1: Bereid Uw Data Voor

Uw data moet in een van deze formaten zijn:
- **CSV** met coördinaten (X/Y of lat/lon kolommen)
- **Shapefile** (.shp met .dbf, .shx, .prj)
- **GeoJSON** (.geojson)
- **Excel** (.xlsx) met coördinaat kolommen
- **NetCDF** (.nc) voor raster/grid data

Plaats het databestand in de juiste `data/` subdirectory.

### Stap 2: Laad de Data

Voeg data-laadcode toe bovenaan `create_comprehensive_map.py` (rond regel 400-600 waar andere data wordt geladen):

#### Voor CSV bestanden:
```python
# Laad uw CSV data
my_data_file = 'data/warmte_data/Uw_Data_Bestand.csv'
if os.path.exists(my_data_file):
    my_df = pd.read_csv(my_data_file)
    print(f"Geladen: {len(my_df)} records uit uw data")
```

#### Voor Shapefiles:
```python
# Laad shapefile
my_shapefile = 'data/uw_folder/uw_data.shp'
if os.path.exists(my_shapefile):
    my_gdf = gpd.read_file(my_shapefile)
    my_gdf = my_gdf.to_crs(epsg=4326)  # Converteer naar WGS84 lat/lon
    print(f"Geladen: {len(my_gdf)} features uit shapefile")
```

#### Voor GeoJSON:
```python
# Laad GeoJSON
my_geojson = 'data/uw_folder/uw_data.geojson'
if os.path.exists(my_geojson):
    my_gdf = gpd.read_file(my_geojson)
    print(f"Geladen: {len(my_gdf)} features uit GeoJSON")
```

### Stap 3: Converteer Coördinaten (indien nodig)

Als uw data Nederlandse RD-coördinaten gebruikt (EPSG:28992), converteer naar WGS84:

```python
from pyproj import Transformer

# Maak transformer van RD naar WGS84
transformer = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)

# Voor DataFrame met X, Y kolommen
my_df['lon'], my_df['lat'] = transformer.transform(
    my_df['X'].values,
    my_df['Y'].values
)
```

### Stap 4: Maak een Feature Group

Voeg een nieuwe FeatureGroup toe voor uw laag (rond regel 1200-1400):

```python
# ============ UW NIEUWE LAAG ============
print("Toevoegen van Uw Nieuwe Laag...")
my_layer_group = folium.FeatureGroup(name='🔶 Uw Laagnaam', show=False)

for idx, row in my_gdf.iterrows():
    # Maak popup HTML
    popup_html = f"""
    <div style="font-family: Arial; width: 280px;">
        <h4 style="color: #FF6600; margin-bottom: 10px;">
            🔶 Uw Data Type
        </h4>
        <table style="width: 100%; font-size: 12px;">
            <tr><td><b>Naam:</b></td><td>{row.get('Naam', 'N/B')}</td></tr>
            <tr><td><b>Waarde:</b></td><td>{row.get('Waarde', 'N/B')}</td></tr>
        </table>
    </div>
    """

    # Voeg marker toe aan de laag
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=6,
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=f"Uw Data: {row.get('Naam', 'N/B')}",
        color='#FF6600',      # Randkleur
        fillColor='#FF9933',  # Vulkleur
        fillOpacity=0.7,
        weight=2
    ).add_to(my_layer_group)

my_layer_group.add_to(m)
print(f"  Toegevoegd: {len(my_gdf)} markers aan Uw Laag")
```

### Stap 5: Opnemen in Score Berekening (Optioneel)

Als uw data de warmtescore moet beïnvloeden, pas de `calculate_location_score()` functie aan:

1. Voeg toe aan `all_warmte_sources` lijst tijdens data laden:
```python
all_warmte_sources.append({
    'lat': row.geometry.y,
    'lon': row.geometry.x,
    'type': 'Uw Data Type',
    'name': row.get('Naam', 'N/B'),
    'color': '#FF6600',
    'uw_waarde': row.get('Waarde', 0),
    'power_display': f"{row.get('Waarde', 0):.1f} MW"
})
```

2. Voeg toe aan score_breakdown in `calculate_location_score()`:
```python
score_breakdown = {
    'mt_warmte_mwth': 0.0,
    'datacenter_vermogen': 0.0,
    'uw_data_waarde': 0.0,  # Voeg dit toe
    'geothermie_heat': 0.0
}
```

3. Voeg verwerkingslogica toe:
```python
elif source['type'] == 'Uw Data Type':
    score_breakdown['uw_data_waarde'] += source.get('uw_waarde', 0) or 0
```

4. Neem op in raw_score berekening:
```python
raw_score = (
    score_breakdown['mt_warmte_mwth'] * 1.0 +
    score_breakdown['datacenter_vermogen'] * 1.0 +
    score_breakdown['uw_data_waarde'] * 0.5 +  # Pas gewicht aan indien nodig
    score_breakdown['geothermie_heat'] * 0.01
)
```

### Stap 6: Werk Documentatie Bij

Voeg uw nieuwe laag toe aan:
- `MANUAL.md` - Gebruikersdocumentatie
- `README.md` - Databronnen tabel

### Voorbeeld: Een Polygoonlaag Toevoegen

Voor polygoon data (zoals zones of gebieden):

```python
# ============ UW POLYGOON LAAG ============
my_polygon_group = folium.FeatureGroup(name='📍 Uw Zones', show=False)

for idx, row in my_zones_gdf.iterrows():
    # Stijl de polygoon
    style = {
        'fillColor': '#FF6600',
        'color': '#CC5500',
        'weight': 2,
        'fillOpacity': 0.3
    }

    popup_html = f"<b>{row.get('ZoneNaam', 'Zone')}</b><br>Oppervlakte: {row.get('Oppervlakte', 'N/B')}"

    folium.GeoJson(
        row.geometry.__geo_interface__,
        style_function=lambda x, s=style: s,
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=row.get('ZoneNaam', 'Zone')
    ).add_to(my_polygon_group)

my_polygon_group.add_to(m)
```

### Voorbeeld: Een Heatmap Laag Toevoegen

Voor dichtheid/intensiteit visualisatie:

```python
from folium.plugins import HeatMap

# Bereid heat data voor: lijst van [lat, lon, intensiteit]
heat_data = [[row.geometry.y, row.geometry.x, row.get('intensiteit', 1)]
             for idx, row in my_gdf.iterrows()]

# Maak heatmap
HeatMap(
    heat_data,
    min_opacity=0.3,
    max_zoom=13,
    radius=15,
    blur=10,
    gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}
).add_to(folium.FeatureGroup(name='🔥 Warmte Intensiteit').add_to(m))
```

---

## Uitgeschakelde Lagen

Sommige lagen zijn uitgecommentarieerd in de code en kunnen opnieuw worden ingeschakeld:

### Condens Warmte (Koelprocessen)

Deze laag is momenteel uitgeschakeld. Om opnieuw in te schakelen:

1. Zoek naar `CONDENS WARMTE DATA COLLECTION` in `create_comprehensive_map.py`
2. Verwijder commentaar van het data-laadblok (regels ~527-612)
3. Zoek naar `CONDENS WARMTE LAYER`
4. Verwijder commentaar van het laag-creatieblok (regels ~1249-1361)
5. Verwijder commentaar van de score-berekeningsregels gemarkeerd met `# DISABLED`

---

## Score Berekening

### Formule

```
ruwe_score = (MT_Warmte_MWth × 1.0) + (Datacenter_Vermogen × 1.0) + (Geothermie_Heat × 0.01)
```

### Normalisatie

```
genormaliseerde_score = 30 × log₁₀(ruwe_score + 1) + 10
```

Deze logaritmische schaling geeft:
- 1 MW ≈ 30 punten
- 10 MW ≈ 60 punten
- 100 MW ≈ 90 punten

### Score Interpretatie

| Score | Label | Betekenis |
|------:|-------|-----------|
| 80+ | Uitstekend | Uitstekend warmtepotentieel |
| 60-79 | Goed | Goed potentieel |
| 40-59 | Matig | Matig potentieel |
| 20-39 | Beperkt | Beperkt potentieel |
| <20 | Minimaal | Minimaal potentieel |

### Besparingsberekening

```
besparing = min(ruwe_score, totaal_verbruik × 0.5)
```

Besparingen zijn begrensd op 50% van het totale energieverbruik van het gebouw.

---

## Output

Het script genereert:
1. **comprehensive_energy_map.html** - Interactieve Folium kaart met alle lagen
2. Console output met data-laadvoortgang en samenvattende statistieken

---

## Probleemoplossing

### Veelvoorkomende Problemen

**"FileNotFoundError: data/..."**
- Controleer of de datastructuur overeenkomt met de verwachte indeling
- Controleer of bestandspaden relatief zijn aan de projectroot

**"CRS transformatie fout"**
- Installeer pyproj: `pip install pyproj`
- Controleer of het bron-CRS correct is gespecificeerd

**"Geheugenfout bij grote datasets"**
- Verminder het aantal punten door te samplen
- Gebruik eenvoudigere geometrieën (centroïden in plaats van polygonen)

**"Kaart laadt niet in browser"**
- Probeer een andere browser (Chrome/Firefox aanbevolen)
- Controleer of het HTML-bestand niet te groot is (>50MB kan problemen geven)

### Prestatie Tips

- Stel `show=False` in voor zware lagen om initiële laadtijd te verbeteren
- Gebruik `simplify()` op complexe geometrieën
- Beperk het aantal markers per laag tot <5000

---

## Bijdragen

1. Fork de repository
2. Maak een feature branch
3. Maak uw wijzigingen
4. Test de kaartgeneratie
5. Dien een pull request in

---

## Licentie

Intern project - neem contact op met projectbeheerders voor gebruiksvoorwaarden.

---

## Voor Niet-Technische Gebruikers

Zie [MANUAL.md](MANUAL.md) voor:
- Hoe de kaart te gebruiken
- Wat de symbolen en kleuren betekenen
- Hoe de scores te interpreteren
- Veelgestelde vragen
