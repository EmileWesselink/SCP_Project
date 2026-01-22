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
