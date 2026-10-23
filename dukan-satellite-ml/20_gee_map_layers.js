// ============================================================
// DUKAN STUDY-AREA MAP: export real layers for the map figure
// Run at code.earthengine.google.com -> Run -> Tasks tab ->
// press RUN on all 5 tasks -> files appear in Drive/Dukan_Map.
// ============================================================

var region = ee.Geometry.Rectangle([44.5, 35.4, 46.8, 37.2]);
Map.centerObject(region, 8);

// 1) Elevation (SRTM, 120 m) for the shaded-relief background
var dem = ee.Image('USGS/SRTMGL1_003').select('elevation').toInt16();
Export.image.toDrive({
  image: dem, description: 'map_dem', folder: 'Dukan_Map',
  region: region, scale: 120, crs: 'EPSG:4326', maxPixels: 1e10
});

// 2) HydroBASINS level 12 (small units, used to trace the exact
//    basin upstream of the dam)
var hb12 = ee.FeatureCollection('WWF/HydroSHEDS/v1/Basins/hybas_12')
  .filterBounds(region);
Export.table.toDrive({
  collection: hb12, description: 'map_hybas12', folder: 'Dukan_Map',
  fileFormat: 'GeoJSON'
});

// 3) The level-7 units used for the climate predictors
//    (documents exactly what the CSV averages cover)
var hb7 = ee.FeatureCollection('WWF/HydroSHEDS/v1/Basins/hybas_7')
  .filterBounds(ee.Geometry.Polygon([
    [[44.9, 36.4], [46.6, 36.9], [46.6, 35.7], [45.0, 35.8]]]));
Export.table.toDrive({
  collection: hb7, description: 'map_hybas7_predictors', folder: 'Dukan_Map',
  fileFormat: 'GeoJSON'
});

// 4) River network (HydroRIVERS / free-flowing rivers)
var rivers = ee.FeatureCollection('WWF/HydroSHEDS/v1/FreeFlowingRivers')
  .filterBounds(region)
  .filter(ee.Filter.gte('UPLAND_SKM', 100));
Export.table.toDrive({
  collection: rivers, description: 'map_rivers', folder: 'Dukan_Map',
  fileFormat: 'GeoJSON'
});

// 5) Dukan Lake outline (JRC water occurrence >= 10%), same box as the study
var box = ee.Geometry.Rectangle([44.80, 35.94, 45.15, 36.22]);
var lakeMask = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
  .select('occurrence').gte(10).selfMask().clip(box).toInt();
var lake = lakeMask.reduceToVectors({
  geometry: box, scale: 30, maxPixels: 1e10, bestEffort: true,
  geometryType: 'polygon', labelProperty: 'water'
});
Export.table.toDrive({
  collection: lake, description: 'map_lake', folder: 'Dukan_Map',
  fileFormat: 'GeoJSON'
});

Map.addLayer(dem, {min: 300, max: 3000}, 'DEM');
Map.addLayer(rivers, {color: '0066cc'}, 'Rivers');
Map.addLayer(lakeMask, {palette: ['blue']}, 'Lake');
