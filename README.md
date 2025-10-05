# NASA-Space-Apps-Project

## master.csv data fields (2020 - 2024 [inclusive]):

### **MERRA-2 Atmospheric Variables:**
- **PRECTOTCORR** (Bias corrected total precipitation [kg m<sup>-2</sup> s<sup>-1</sup>]) → **precipitation** [mm/hour]
- **T2M** (2-meter air temperature [K]) → **temperature** [°C]
- **QV2M** (2-meter specific humidity [kg/kg]) → **humidity** [g/kg]
- **PS** (Surface pressure [Pa]) → **surface_pressure** [hPa/mbar]
- **U10M** (10-meter U-wind component [m/s]) → **u_wind** [m/s]
- **V10M** (10-meter V-wind component [m/s]) → **v_wind** [m/s]

### **Climate Indices (External Sources):**
- **SST** (Sea Surface Temperature [°C]) - *Source: NOAA OISST v2.1, Niño 3.4 region*
- **IOD** (Indian Ocean Dipole Index) - *Source: Australian Bureau of Meteorology*
- **NAO** (North Atlantic Oscillation Index) - *Source: NOAA PSL*
- **AO** (Arctic Oscillation Index) - *Source: NOAA PSL*

## Data Processing:

### **MERRA-2 Processing:**
- **Source**: NASA MERRA-2 atmospheric reanalysis
- **Temporal Resolution**: Hourly data aggregated to daily
- **Spatial Coverage**: 25 cities across South Asia
- **Aggregation Methods**:
  - Precipitation: Daily sum
  - Temperature, Humidity, Pressure, Winds: Daily mean
- **Unit Conversions Applied**:
  - Precipitation: kg/m²/s → mm/hour
  - Temperature: Kelvin → Celsius  
  - Humidity: kg/kg → g/kg
  - Pressure: Pa → hPa

### **Climate Indices Processing:**
- **SST**: Daily resolution from NOAA OISST v2.1 (Niño 3.4 region)
- **IOD**: Weekly resolution from BOM, converted to daily using climatological interpolation
- **NAO**: Daily resolution from NOAA PSL
- **AO**: Monthly resolution from NOAA PSL, converted to daily using climatological interpolation

### **Data Integration:**
- **Temporal Coverage**: January 1, 2020 - December 31, 2024
- **Spatial Integration**: Climate indices applied uniformly across all 25 regions
- **Gap Handling**: Linear interpolation for small data gaps
- **Validation**: Comprehensive missing value analysis and statistical validation

## Data Sources:
- **MERRA-2**: NASA Global Modeling and Assimilation Office
- **SST**: NOAA Physical Sciences Laboratory (OISST v2.1)
- **IOD**: Australian Bureau of Meteorology  
- **NAO/AO**: NOAA Climate Prediction Center

## Webiste Frontend:
https://github.com/fasihh/NASAFrontEnd

## Website Backend:
https://github.com/fasihh/NASABackend 
