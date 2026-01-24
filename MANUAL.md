# Handleiding - RVB Warmtepotentie Kaart

## Wat is deze kaart?

Deze interactieve kaart laat zien welke overheidsgebouwen (RVB-panden) het meeste kunnen besparen door gebruik te maken van warmtebronnen in de buurt. Denk aan restwarmte van datacenters, industriele processen of aardwarmte.

De kaart helpt bij het beantwoorden van de vraag: **"Welke gebouwen kunnen bijdragen aan het verminderen van netcongestie door lokale warmtebronnen te gebruiken?"**

---

## Hoe gebruik je de kaart?

### De kaart openen

1. Open het bestand `comprehensive_energy_map.html` in uw webbrowser
2. De kaart laadt automatisch met alle beschikbare lagen

### Navigeren

- **Inzoomen**: Scroll met uw muiswiel of gebruik de + knop linksboven
- **Uitzoomen**: Scroll terug of gebruik de - knop
- **Verplaatsen**: Klik en sleep de kaart

### Lagen aan/uit zetten

Rechtsboven in de kaart vindt u het **lagenmenu** (pictogram met gestapelde vierkantjes). Hier kunt u:
- Verschillende kaartlagen aan- of uitzetten door het vakje aan te vinken
- De achtergrondkaart wijzigen (licht, donker of stratenkaart)

---

## Wat betekenen de symbolen?

### RVB Gebouwen (driehoekjes)

Elk driehoekje op de kaart is een RVB-gebouw. De kleur geeft aan hoe goed de locatie scoort voor warmtepotentie:

| Kleur | Betekenis |
|-------|-----------|
| 🟢 Groen | Uitstekend - Veel warmtebronnen in de buurt |
| 🔵 Blauw | Goed - Goede warmtepotentie |
| 🟡 Geel | Matig - Redelijke warmtepotentie |
| 🟠 Oranje | Beperkt - Weinig warmtebronnen |
| 🔴 Rood | Minimaal - Nauwelijks warmtebronnen beschikbaar |

### Warmtebronnen (cirkels)

Op de kaart ziet u ook verschillende warmtebronnen als gekleurde cirkels:

| Symbool | Type | Wat is het? |
|---------|------|-------------|
| 🔴 Rode cirkel | MT Warmte | Midden-temperatuur warmtebronnen (industrie) |
| 💻 Blauwe cirkel | Datacenter | Restwarmte van datacenters |
| 🌋 Oranje vlak | Geothermie | Aardwarmte potentie (heatmap) |

### Defensie locaties (speciale markers)

Defensie-locaties worden apart weergegeven met:
- **Paarse markers**: Bovenregionale defensielocaties
- **Oranje markers**: Locatiespecifieke defensie-faciliteiten

---

## Gebouwinformatie bekijken

Klik op een gebouw (driehoekje) om een popup te openen met gedetailleerde informatie:

### Basisgegevens

- **Objectnaam**: Naam van het gebouw
- **Adres**: Locatie van het gebouw
- **Eigenaar**: Wie het gebouw bezit
- **Bouwwerkfunctie**: Waar het gebouw voor wordt gebruikt

### Warmte Score

De warmtescore is een getal van 0 tot 100 dat aangeeft hoeveel warmte-besparingspotentieel het gebouw heeft:

| Score | Beoordeling | Wat betekent dit? |
|------:|-------------|-------------------|
| 80-100 | Uitstekend | Veel warmtebronnen dichtbij, groot besparingspotentieel |
| 60-79 | Goed | Goede mogelijkheden voor warmtebenutting |
| 40-59 | Matig | Redelijke mogelijkheden |
| 20-39 | Beperkt | Weinig warmtebronnen beschikbaar |
| 0-19 | Minimaal | Nauwelijks warmtebronnen in de buurt |

### Verbruik Oordeel

Het oordeel geeft aan hoe het huidige energieverbruik zich verhoudt tot de contractcapaciteit:

| Kleur | Betekenis |
|-------|-----------|
| 🟢 Groen | Goed - Minder dan 80% van capaciteit gebruikt |
| 🟠 Oranje | Waarschuwing - 80-100% van capaciteit gebruikt |
| 🔴 Rood | Kritiek - Meer dan 100% van capaciteit (overschrijding) |

### Warmtenet Status

Bij elk gebouw ziet u of het op een bestaand warmtenet is aangesloten:
- **Ja**: Het gebouw ligt binnen een warmtenetgebied
- **Nee**: Het gebouw ligt buiten bestaande warmtenetten

### Nabijgelegen Warmtebronnen

Onderaan de popup vindt u een tabel met alle warmtebronnen binnen 1 kilometer, inclusief:
- Type warmtebron
- Naam
- Vermogen of energie
- Afstand tot het gebouw

---

## De Score Uitgelegd

De warmtescore wordt berekend op basis van warmtebronnen binnen 1 kilometer van het gebouw:

### Componenten

| Component | Wat is het? | Eenheid |
|-----------|-------------|---------|
| MT Warmte | Midden-temperatuur industriele warmte | MW (megawatt) |
| Datacenter | Restwarmte van datacenters (alleen >60°C) | MW |
| Geothermie | Aardwarmte potentie (alleen voor Defensie) | - |

### Hoe werkt de berekening?

1. **Ruwe score**: Alle warmtebronnen binnen 1 km worden opgeteld
2. **Normalisatie**: De ruwe score wordt omgezet naar een schaal van 0-100
3. **Logaritmische schaal**: Dit zorgt ervoor dat zowel kleine als grote hoeveelheden warmte goed worden weergegeven

### Besparing

De potentiele besparing wordt berekend als:
- Maximaal 50% van het totale energieverbruik van het gebouw
- Afhankelijk van de beschikbare warmtebronnen in de buurt

---

## Dashboard Panelen

### Top 10 Potentiele Groei

Rechtsonder in de kaart vindt u een panel met de 10 gebouwen met het hoogste besparingspotentieel. Klik op een gebouw in de lijst om ernaar toe te navigeren.

### Legenda

Linksonder vindt u de legenda die uitlegt wat de verschillende kleuren en symbolen betekenen.

---

## Beschikbare Kaartlagen

### Achtergrondkaarten
- **Light Map** - Lichte, overzichtelijke achtergrond (standaard)
- **Street Map** - Gedetailleerde OpenStreetMap stijl
- **Dark Map** - Donkere achtergrond (beter voor details)

### Gebouwlagen
- **RVB Buildings** - Alle RVB-panden met warmtescore
- **Defensie Locaties** - Defensie-specifieke locaties

### Warmtebronnen
- **Warmte Bronnen (MT)** - Midden-temperatuur industriele warmtebronnen
- **Datacenter Warmte** - Restwarmte van datacenters
- **PDOK Restwarmte** - Industriele restwarmte via overheidsdata
- **ThermoGIS Geothermie** - Aardwarmte potentie (heatmap weergave)

<!-- - **Condens Warmte (Koelprocessen)** - Condenswarmte uit koelprocessen (DISABLED - uncomment to re-enable) -->

### Infrastructuur
- **Warmte Net Areas** - Bestaande warmtenetgebieden
- **Netcongestie Gebieden** - Gebieden met netwerk-congestie
- **Netherlands Boundary** - Landsgrens Nederland

---

## Veelgestelde Vragen

### Waarom zie ik sommige gebouwen niet?
Niet alle gebouwen hebben volledige data. Gebouwen zonder coordinaten of met ontbrekende gegevens worden niet weergegeven.

### Wat betekent "Op bestaand warmtenet: Ja"?
Dit betekent dat het gebouw binnen een gebied ligt waar al een warmtenet aanwezig is. Dit kan de implementatie van warmtebenutting vergemakkelijken.

### Hoe actueel is de data?
De data is afkomstig uit verschillende bronnen met verschillende update-frequenties. Raadpleeg de README voor specifieke bronvermeldingen.

### Kan ik de kaart offline gebruiken?
Ja, het HTML-bestand kan lokaal worden geopend. Sommige achtergrondkaarten vereisen echter een internetverbinding.

---

## Technische Ondersteuning

Voor technische vragen of problemen, raadpleeg:
- De [README.md](README.md) voor installatie en configuratie
- Het projectteam voor data-gerelateerde vragen

---

## Data Bronnen

| Data | Bron |
|------|------|
| RVB Locaties | Rijksvastgoedbedrijf dataset |
| MT Warmte | PBL - Startanalyse warmtebronnen (2024) |
| Datacenter Warmte | RVO - DataCentra Warmte data |
| PDOK Restwarmte | PDOK WFS - Overheids geodata service |
| Geothermie | ThermoGIS - Aardwarmte potentie data |
| Warmtenetten | CBS Buurtkaart 2020 |
| Netcongestie | Liander congestiegebieden |
<!-- | Condens Warmte | RVO - CondensWarmte uit Koelprocessen | (DISABLED) -->
