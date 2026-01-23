# SCP Project - RVB Warmtepotentie Kaart

Een interactieve kaart die het warmte-besparingspotentieel van RVB (Rijksvastgoedbedrijf) locaties visualiseert op basis van nabijgelegen warmtebronnen.

## Overzicht

Deze kaart combineert gegevens van RVB-locaties met diverse warmtebronnen om te analyseren hoe gebouwen kunnen bijdragen aan de vermindering van netcongestie door gebruik te maken van lokale warmtebronnen.

## Kaartlagen

De kaart bevat de volgende lagen (in/uit te schakelen via het lagenmenu):

### Basislagen
- **Light Map** - Lichte achtergrondkaart (standaard)
- **Street Map** - OpenStreetMap stijl
- **Dark Map** - Donkere achtergrondkaart
- **Netherlands Boundary** - Landsgrens Nederland

### RVB Locaties
- **RVB Buildings** - Alle RVB-panden (driehoek markers)
- **Defensie Locaties** - Defensie-specifieke locaties (oranje markers)

### Warmtebronnen
- **Warmte Bronnen (MT)** - Midden-temperatuur warmtebronnen uit de PBL startanalyse
- **Datacenter Warmte** - Restwarmte van datacenters
- **PDOK Restwarmte (Industrie)** - Industriele restwarmte via PDOK WFS
- **Condens Warmte (Koelprocessen)** - Condenswarmte uit koelprocessen
- **ThermoGIS Geothermie** - Aardwarmte potentie (heatmap)

### Analyse
- **Netcongestie Gebieden** - Gebieden met netcongestie (rood/oranje)

## Popup Informatie

Klik op een RVB-locatie om gedetailleerde informatie te zien:

### Basisinformatie
- Objectnaam en adres
- Eigenaar en bouwwerkfunctie
- Contractcapaciteit en maximaal verbruik

### Warmte Score ("Mogelijke Besparing Warmte Opwek")

De score geeft aan hoeveel warmte-besparingspotentie een locatie heeft op basis van nabijgelegen warmtebronnen (< 1 km). De score bestaat uit:

| Component | Bron | Eenheid |
|-----------|------|---------|
| MT Warmte | PBL Startanalyse | MW thermisch |
| Datacenter (>60°C) | Datacenters met temperatuur > 60°C | MW |
| PDOK Restwarmte | Industriele restwarmte (PDOK) | TJ |
| Condens Warmte | Koelprocessen | TJ |
| Geothermie | ThermoGIS (alleen Defensie) | - |

De ruwe score wordt omgezet naar een genormaliseerde schaal (0-100):
- **0-20**: Zeer Laag - nauwelijks warmtebronnen beschikbaar
- **20-40**: Laag - beperkte warmtebronnen
- **40-60**: Gemiddeld - redelijke warmtepotentie
- **60-80**: Hoog - goede warmtepotentie
- **80-100**: Zeer Hoog - uitstekende warmtepotentie

### Verbruik Oordeel

Het oordeel wordt bepaald door de verhouding tussen maximaal verbruik en contractcapaciteit:
- **Groen**: ≤ 80% van contractcapaciteit benut
- **Oranje**: 80-100% van contractcapaciteit benut
- **Rood**: > 100% van contractcapaciteit benut
- **Onbekend**: Geen verbruiksgegevens beschikbaar

### Nieuw Totaal Verbruik

Laat zien hoe het verbruik zou verbeteren met warmtebesparing:
```
Nieuw Totaal Verbruik = Totaal Verbruik - Besparing Warmte Opwek
```

De popup toont twee oordeel-kleuren:
- **HUIDIG**: Het huidige verbruiksoordeel
- **MET WARMTE**: Het oordeel na toepassing van warmtebesparing

### Nabijgelegen Warmtebronnen

Een overzichtstabel toont alle warmtebronnen binnen 1 km met:
- Type warmtebron
- Naam van de bron
- Vermogen/energie capaciteit
- Afstand tot de RVB-locatie

### Score implementatie

  raw_score = (MT_Warmte_MWth × 1.0) + (Datacenter_Vermogen × 1.0) + (Condens_Warmte_TJ × 0.1) + (Geothermie_Heat ×
  0.01)

  Components (from nearby heat sources within ~5km):
  - MT Warmte: Thermal power in MW (weight: 1.0)
  - Datacenter Vermogen: Datacenter capacity in MW (weight: 1.0)
  - Condens Warmte: Cooling process waste heat in TJ (weight: 0.1 to convert TJ→MW equivalent)
  - Geothermie: Only for Defensie locations (weight: 0.01, scaled down)

  Normalized Score (0-100) (line 212)

  normalized = 30 × log₁₀(raw_score + 1) + 10

  This logarithmic scaling means:
  - 1 MW ≈ 30 points
  - 10 MW ≈ 60 points
  - 100 MW ≈ 90 points

| Score | Label     | Meaning                    |
|------:|-----------|----------------------------|
| 80+   | Uitstekend | Excellent heat potential   |
| 60–79 | Goed       | Good potential             |
| 40–59 | Matig      | Moderate potential         |
| 20–39 | Beperkt    | Limited potential          |
| <20   | Minimaal   | Minimal potential          |
  Besparing (Savings) Calculation (lines 694-698)

  besparing = min(raw_score, totaal_verbruik × 0.5)

  The savings is capped at 50% of the building's total energy consumption.

## Data Bronnen

| Data | Bron |
|------|------|
| RVB Locaties | RVB dataset (punten en polygonen) |
| MT Warmte | PBL - Download-MT-Warmtebronnen startanalyse (2024) |
| Datacenter Warmte | RVO - Download-LT DataCentraWarmte |
| Condens Warmte | RVO - Download-LT CondensWarmte uit Koelprocessen |
| PDOK Restwarmte | PDOK WFS - service.pdok.nl/rvo/restwarmte |
| Geothermie | ThermoGIS NetCDF - OVERVIEW_potential_recoverable_heat |
| Netcongestie | Liander dataset |

## Gebruik

1. Voer het script uit: `python create_comprehensive_map.py`
2. Open het gegenereerde HTML-bestand: `comprehensive_energy_map.html`
3. Gebruik het lagenmenu (rechtsboven) om lagen aan/uit te zetten
4. Klik op RVB-locaties voor gedetailleerde warmte-analyse

## Vereisten

- Python 3.x
- folium
- geopandas
- pandas
- numpy
- requests
- netCDF4

Installeer met: `pip install folium geopandas pandas numpy requests netCDF4`
