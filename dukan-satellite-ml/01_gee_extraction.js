// ============================================================
// DUKAN ML STUDY - GEE EXTRACTION v2 (FINAL)
// Improvements over v1:
//  - Reservoir mask auto-derived from JRC Global Surface Water
//    (no manual polygon errors)
//  - Lesser Zab catchment auto-selected from HydroSHEDS in GEE
//  - Sentinel-1 SAR water extent added (2015+, cloud-independent)
//  - JRC monthly water history added as independent cross-check
// Run in: code.earthengine.google.com | Runtime: exports ~10-30 min
// ============================================================

var START = '2011-01-01';
var END   = '2026-01-01';

// ---------- 1. AUTO RESERVOIR MASK (JRC) ----------
// Loose box around Dukan lake only (not the river downstream)
var box = ee.Geometry.Rectangle([44.80, 35.85, 45.15, 36.20]);

// JRC water occurrence: pixels that were water >= 10% of 1984-2021
var jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater');
var reservoirMask = jrc.select('occurrence').gte(10).clip(box).selfMask();

// Vectorize once for reduceRegion geometry
var reservoir = reservoirMask.reduceToVectors({
  geometry: box, scale: 90, maxPixels: 1e10, bestEffort: true
}).geometry();

Map.centerObject(box, 10);
Map.addLayer(reservoirMask, {palette: ['blue']}, 'Reservoir mask (JRC>=10%)');

// ---------- 2. AUTO CATCHMENT (HydroSHEDS level 7) ----------
// Lesser Zab upstream basins: select all hybas_7 polygons whose
// centroid falls in the upper Lesser Zab region (Iraq+Iran headwaters)
var hybas = ee.FeatureCollection('WWF/HydroSHEDS/v1/Basins/hybas_7');
var upperZabRegion = ee.Geometry.Polygon([
  [[44.9, 36.4], [46.6, 36.9], [46.6, 35.7], [45.0, 35.8]]
]); // covers headwaters in Iran (Sardasht/Piranshahr) + KRI portion
var catchment = hybas.filterBounds(upperZabRegion).geometry();
Map.addLayer(catchment, {color: 'green'}, 'Catchment (verify visually!)', false);
// >>> VERIFY on the map that this covers the Lesser Zab basin upstream
// >>> of Dukan and nothing else. Adjust upperZabRegion corners if needed.

// Snow zone = catchment area above 1500 m (where snow actually falls)
var dem = ee.Image('USGS/SRTMGL1_003');
var snowZone = dem.gte(1500).selfMask().clip(catchment);

// ---------- 3. MONTH LIST ----------
var months = ee.List.sequence(0,
  ee.Date(END).difference(ee.Date(START), 'month').subtract(1))
  .map(function(m){ return ee.Date(START).advance(m, 'month'); });

// ---------- 4. LANDSAT WATER AREA ----------
function maskL57(img){
  var qa = img.select('QA_PIXEL');
  var m = qa.bitwiseAnd(1<<3).eq(0).and(qa.bitwiseAnd(1<<4).eq(0));
  return img.updateMask(m)
    .select(['SR_B2','SR_B5'], ['GREEN','SWIR'])
    .multiply(0.0000275).add(-0.2)
    .copyProperties(img, ['system:time_start']);
}
function maskL89(img){
  var qa = img.select('QA_PIXEL');
  var m = qa.bitwiseAnd(1<<3).eq(0).and(qa.bitwiseAnd(1<<4).eq(0));
  return img.updateMask(m)
    .select(['SR_B3','SR_B6'], ['GREEN','SWIR'])
    .multiply(0.0000275).add(-0.2)
    .copyProperties(img, ['system:time_start']);
}
var landsat = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').map(maskL57)
  .merge(ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').map(maskL57))
  .merge(ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').map(maskL89))
  .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').map(maskL89))
  .filterDate(START, END).filterBounds(box);

var waterArea = ee.FeatureCollection(months.map(function(d){
  d = ee.Date(d);
  var col = landsat.filterDate(d, d.advance(1,'month'));
  var n = col.size();
  var mndwi = ee.Image(ee.Algorithms.If(n.gt(0),
    col.median().normalizedDifference(['GREEN','SWIR']),
    ee.Image.constant(-999))).rename('MNDWI');
  var area = mndwi.gt(0).updateMask(reservoirMask)
    .multiply(ee.Image.pixelArea()).divide(1e6)
    .reduceRegion({reducer: ee.Reducer.sum(), geometry: reservoir,
                   scale: 30, maxPixels: 1e10}).get('MNDWI');
  return ee.Feature(null, {date: d.format('YYYY-MM'),
    landsat_area_km2: area, n_landsat: n});
}));
Export.table.toDrive({collection: waterArea,
  description: 'Dukan_landsat_area', folder: 'Dukan_ML', fileFormat: 'CSV'});

// ---------- 5. SENTINEL-1 SAR WATER AREA (2015+) ----------
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterDate('2015-01-01', END).filterBounds(box)
  .filter(ee.Filter.eq('instrumentMode','IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'))
  .select('VV');

var s1Area = ee.FeatureCollection(months.map(function(d){
  d = ee.Date(d);
  var col = s1.filterDate(d, d.advance(1,'month'));
  var n = col.size();
  var vv = ee.Image(ee.Algorithms.If(n.gt(0), col.median(),
    ee.Image.constant(999))).rename('VV');
  // Water = low backscatter; -16 dB threshold works well on lakes
  var area = vv.lt(-16).updateMask(reservoirMask)
    .multiply(ee.Image.pixelArea()).divide(1e6)
    .reduceRegion({reducer: ee.Reducer.sum(), geometry: reservoir,
                   scale: 30, maxPixels: 1e10}).get('VV');
  return ee.Feature(null, {date: d.format('YYYY-MM'),
    s1_area_km2: area, n_s1: n});
}));
Export.table.toDrive({collection: s1Area,
  description: 'Dukan_s1_area', folder: 'Dukan_ML', fileFormat: 'CSV'});

// ---------- 6. JRC MONTHLY WATER (independent cross-check, to 2021) ----------
var jrcMonthly = ee.ImageCollection('JRC/GSW1_4/MonthlyHistory')
  .filterDate(START, '2022-01-01');
var jrcArea = ee.FeatureCollection(jrcMonthly.map(function(img){
  var area = img.eq(2).multiply(ee.Image.pixelArea()).divide(1e6)
    .reduceRegion({reducer: ee.Reducer.sum(), geometry: reservoir,
                   scale: 30, maxPixels: 1e10}).get('water');
  return ee.Feature(null, {
    date: ee.Date(img.get('system:time_start')).format('YYYY-MM'),
    jrc_area_km2: area});
}));
Export.table.toDrive({collection: jrcArea,
  description: 'Dukan_jrc_area', folder: 'Dukan_ML', fileFormat: 'CSV'});

// ---------- 7. RAINFALL (CHIRPS) ----------
var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/PENTAD').filterDate(START, END);
var rain = ee.FeatureCollection(months.map(function(d){
  d = ee.Date(d);
  var mm = chirps.filterDate(d, d.advance(1,'month')).sum()
    .reduceRegion({reducer: ee.Reducer.mean(), geometry: catchment,
                   scale: 5000, maxPixels: 1e10}).get('precipitation');
  return ee.Feature(null, {date: d.format('YYYY-MM'), rain_mm: mm});
}));
Export.table.toDrive({collection: rain,
  description: 'Dukan_rain', folder: 'Dukan_ML', fileFormat: 'CSV'});

// ---------- 8. ERA5-LAND CLIMATE ----------
var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
  .filterDate(START, END)
  .select(['temperature_2m','potential_evaporation_sum','snowmelt_sum']);
var climate = ee.FeatureCollection(era5.map(function(img){
  var s = img.reduceRegion({reducer: ee.Reducer.mean(), geometry: catchment,
                            scale: 10000, maxPixels: 1e10});
  return ee.Feature(null, {
    date: ee.Date(img.get('system:time_start')).format('YYYY-MM'),
    temp_K: s.get('temperature_2m'),
    pot_evap_m: s.get('potential_evaporation_sum'),
    snowmelt_m: s.get('snowmelt_sum')});
}));
Export.table.toDrive({collection: climate,
  description: 'Dukan_climate', folder: 'Dukan_ML', fileFormat: 'CSV'});

// ---------- 9. MODIS SNOW COVER (elevation-masked) ----------
var modis = ee.ImageCollection('MODIS/061/MOD10A1')
  .filterDate(START, END).select('NDSI_Snow_Cover');
var snow = ee.FeatureCollection(months.map(function(d){
  d = ee.Date(d);
  var frac = modis.filterDate(d, d.advance(1,'month')).mean().gte(40)
    .updateMask(snowZone)
    .reduceRegion({reducer: ee.Reducer.mean(), geometry: catchment,
                   scale: 500, maxPixels: 1e10}).get('NDSI_Snow_Cover');
  return ee.Feature(null, {date: d.format('YYYY-MM'), snow_frac: frac});
}));
Export.table.toDrive({collection: snow,
  description: 'Dukan_snow', folder: 'Dukan_ML', fileFormat: 'CSV'});

// ============================================================
// TO RUN: paste all -> Run -> open Tasks tab -> click RUN on all
// 6 exports -> CSVs appear in Drive/Dukan_ML within ~30 min.
// Then download all 6 CSVs and send them to Claude.
// ============================================================
