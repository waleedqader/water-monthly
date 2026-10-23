// ============================================================
// DUKAN STUDY-AREA MAP — LAYER EXPORT FOR ARCGIS PRO
// Run in code.earthengine.google.com. Exports 4 shapefiles +
// 1 DEM GeoTIFF to Drive/Dukan_Map. Style them in ArcGIS Pro.
// ============================================================

// ---- Reuse the SAME geometries you already verified ----
var box = ee.Geometry.Rectangle([44.80, 35.85, 45.15, 36.20]);
var jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater');
var reservoirMask = jrc.select('occurrence').gte(10).clip(box).selfMask();

// 1) RESERVOIR POLYGON (the lake outline for your map)
var reservoirVec = reservoirMask.reduceToVectors({
  geometry: box, scale: 30, maxPixels: 1e10, bestEffort: true,
  geometryType: 'polygon'
});
Export.table.toDrive({collection: reservoirVec,
  description: 'map_reservoir_polygon', folder: 'Dukan_Map', fileFormat: 'SHP'});

// 2) CATCHMENT POLYGON (Lesser Zab upstream of Dukan)
var hybas = ee.FeatureCollection('WWF/HydroSHEDS/v1/Basins/hybas_7');
var upperZabRegion = ee.Geometry.Polygon([
  [[44.9, 36.4],[46.6, 36.9],[46.6, 35.7],[45.0, 35.8]]]);
var catchment = ee.FeatureCollection([
  ee.Feature(hybas.filterBounds(upperZabRegion).geometry(), {name: 'Lesser Zab catchment'})]);
Export.table.toDrive({collection: catchment,
  description: 'map_catchment_polygon', folder: 'Dukan_Map', fileFormat: 'SHP'});

// 3) RIVER NETWORK (HydroRIVERS, clipped to catchment)
var rivers = ee.FeatureCollection('WWF/HydroSHEDS/v1/FreeFlowingRivers')
  .filterBounds(catchment.geometry());
Export.table.toDrive({collection: rivers,
  description: 'map_rivers', folder: 'Dukan_Map', fileFormat: 'SHP'});

// 4) SNOW ZONE (>1500 m within catchment) as polygon
var dem = ee.Image('USGS/SRTMGL1_003');
var snowPoly = dem.gte(1500).selfMask().clip(catchment)
  .reduceToVectors({geometry: catchment.geometry(), scale: 500,
    maxPixels: 1e10, bestEffort: true, geometryType: 'polygon'});
Export.table.toDrive({collection: snowPoly,
  description: 'map_snowzone_1500m', folder: 'Dukan_Map', fileFormat: 'SHP'});

// 5) DEM GeoTIFF (for hillshade / elevation background in ArcGIS)
Export.image.toDrive({image: dem.clip(catchment.geometry().bounds()),
  description: 'map_DEM_srtm', folder: 'Dukan_Map',
  scale: 90, region: catchment.geometry().bounds(), maxPixels: 1e10});

// ============================================================
// IN ARCGIS PRO (≈45 min once these 5 files are in Drive):
// 1. Add all 5 layers. Set project CRS to WGS84 / UTM 38N.
// 2. DEM → Hillshade (Spatial Analyst) as background.
// 3. Style: catchment = hollow black outline; reservoir = solid blue;
//    rivers = thin blue lines; snow zone = 40% transparent white/purple.
// 4. Add Iran–Iraq border (Living Atlas "World Countries") + star
//    markers for Sardasht (~36.19N,45.40E) & Kani Sib — verify in Google Earth.
// 5. Insert → Layout (A4 landscape): map frame + Iraq inset w/ red box,
//    north arrow, scale bar, legend, graticule.
// 6. Share → Export Layout → PNG 300 dpi → name: fig_studyarea.png
// ============================================================
