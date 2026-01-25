# User Manual - RVB Heat Potential Map

## What is this map?

This interactive map shows which government buildings (RVB properties) can save the most energy by utilizing nearby heat sources. Think of waste heat from datacenters, industrial processes, or geothermal energy.

The map helps answer the question: **"Which buildings can contribute to reducing grid congestion by using local heat sources?"**

---

## How to use the map?

### Opening the map

1. Open the file `comprehensive_energy_map.html` in your web browser
2. The map loads automatically with all available layers

### Navigating

- **Zoom in**: Scroll with your mouse wheel or use the + button in the top left
- **Zoom out**: Scroll back or use the - button
- **Pan**: Click and drag the map

### Toggling layers on/off

In the top right of the map you'll find the **layer menu** (icon with stacked squares). Here you can:
- Toggle different map layers on or off by checking the box
- Change the background map (light, dark, or street map)

---

## What do the symbols mean?

### RVB Buildings (triangles)

Each triangle on the map is an RVB building. The color indicates how well the location scores for heat potential:

| Color | Meaning |
|-------|---------|
| 🟢 Green | Excellent - Many heat sources nearby |
| 🔵 Blue | Good - Good heat potential |
| 🟡 Yellow | Moderate - Reasonable heat potential |
| 🟠 Orange | Limited - Few heat sources |
| 🔴 Red | Minimal - Hardly any heat sources available |

### Heat Sources (circles)

On the map you'll also see various heat sources as colored circles:

| Symbol | Type | What is it? |
|--------|------|-------------|
| 🔴 Red circle | MT Warmte | Medium-temperature heat sources (industrial) |
| 💻 Blue circle | Datacenter | Waste heat from datacenters |
| 🌋 Orange area | Geothermal | Geothermal heat potential (heatmap) |

### Defensie locations (special markers)

Defensie locations are displayed separately with:
- **Purple markers**: Supra-regional defense locations
- **Orange markers**: Location-specific defense facilities

---

## Viewing building information

Click on a building (triangle) to open a popup with detailed information:

### Basic information

- **Object name**: Name of the building
- **Address**: Location of the building
- **Owner**: Who owns the building
- **Building function**: What the building is used for

### Heat Score

The heat score is a number from 0 to 100 indicating how much heat savings potential the building has:

| Score | Rating | What does this mean? |
|------:|--------|----------------------|
| 80-100 | Excellent | Many heat sources nearby, great savings potential |
| 60-79 | Good | Good opportunities for heat utilization |
| 40-59 | Moderate | Reasonable opportunities |
| 20-39 | Limited | Few heat sources available |
| 0-19 | Minimal | Hardly any heat sources nearby |

### Consumption Rating

The rating indicates how current energy consumption compares to contract capacity:

| Color | Meaning |
|-------|---------|
| 🟢 Green | Good - Less than 80% of capacity used |
| 🟠 Orange | Warning - 80-100% of capacity used |
| 🔴 Red | Critical - More than 100% of capacity (exceeded) |

### Heat Network Status

For each building you'll see if it's connected to an existing heat network:
- **Yes**: The building is within a heat network area
- **No**: The building is outside existing heat networks

### Nearby Heat Sources

At the bottom of the popup you'll find a table with all heat sources within 1 kilometer, including:
- Type of heat source
- Name
- Power or energy
- Distance to the building

---

## The Score Explained

The heat score is calculated based on heat sources within 1 kilometer of the building:

### Components

| Component | What is it? | Unit |
|-----------|-------------|------|
| MT Warmte | Medium-temperature industrial heat | MW (megawatt) |
| Datacenter | Waste heat from datacenters (only >60°C) | MW |
| Geothermal | Geothermal heat potential (Defensie only) | - |

### How does the calculation work?

1. **Raw score**: All heat sources within 1 km are summed up
2. **Normalization**: The raw score is converted to a scale of 0-100
3. **Logarithmic scale**: This ensures both small and large amounts of heat are displayed properly

### Savings

The potential savings are calculated as:
- Maximum 50% of the building's total energy consumption
- Depending on available heat sources nearby

---

## Dashboard Panels

### Top 10 Potential Growth

In the bottom right of the map you'll find a panel with the 10 buildings with the highest savings potential. Click on a building in the list to navigate to it.

### Legend

In the bottom left you'll find the legend explaining what the different colors and symbols mean.

---

## Available Map Layers

### Background maps
- **Light Map** - Light, clear background (default)
- **Street Map** - Detailed OpenStreetMap style
- **Dark Map** - Dark background (better for details)

### Building layers
- **RVB Buildings** - All RVB properties with heat score
- **Defensie Locations** - Defensie-specific locations

### Heat sources
- **Heat Sources (MT)** - Medium-temperature industrial heat sources
- **Datacenter Heat** - Waste heat from datacenters
- **PDOK Waste Heat** - Industrial waste heat via government data
- **ThermoGIS Geothermal** - Geothermal heat potential (heatmap display)

### Infrastructure
- **Heat Network Areas** - Existing heat network areas
- **Grid Congestion Areas** - Areas with network congestion
- **Netherlands Boundary** - Country boundary of the Netherlands

---

## Frequently Asked Questions

### Why don't I see some buildings?
Not all buildings have complete data. Buildings without coordinates or with missing information are not displayed.

### What does "On existing heat network: Yes" mean?
This means the building is within an area where a heat network already exists. This can facilitate the implementation of heat utilization.

### How current is the data?
The data comes from various sources with different update frequencies. Consult the README for specific source references.

### Can I use the map offline?
Yes, the HTML file can be opened locally. However, some background maps require an internet connection.

---

## Technical Support

For technical questions or issues, consult:
- The [README.md](README.md) for installation and configuration
- The project team for data-related questions

---

## Data Sources

| Data | Source |
|------|--------|
| RVB Locations | Rijksvastgoedbedrijf dataset |
| MT Warmte | PBL - Heat sources start analysis (2024) |
| Datacenter Heat | RVO - DataCentra Heat data |
| PDOK Waste Heat | PDOK WFS - Government geodata service |
| Geothermal | ThermoGIS - Geothermal potential data |
| Heat Networks | CBS Buurtkaart 2020 |
| Grid Congestion | Liander congestion areas |


## License

Unless stated otherwise, this project is licensed under the CC BY-NC-SA 4.0 License. See the LICENSE file for details.
