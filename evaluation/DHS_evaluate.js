var points = ee.FeatureCollection("users/kakooeimohammd/Africa_IWI_Rural_year_country"),
    imageF1 = ee.Image("users/kakooeimohammd/AfricaLULC/testImage"),
    imageVisParam = {"opacity":1,"bands":["remapped"],"min":0,"max":2,"palette":["ffffff","ffed19","ff39e7"]},
    LSIB = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017");

Map.addLayer(imageF1.select('b4').remap([0,1,2,3,4,5,6,7],[0,0,0,0,0,0,1,2]), imageVisParam, 'Prediction2020', false); 

var img_smod2015 = ee.Image("JRC/GHSL/P2023A/GHS_SMOD/2015").remap([11,12,13,21,22,23,30,-200,10],[0,1,1,2,2,2,2,0,0]).rename(['SMOD2015']);

var img_smod2020 = ee.Image("JRC/GHSL/P2023A/GHS_SMOD/2020").remap([11,12,13,21,22,23,30,-200,10],[0,1,1,2,2,2,2,0,0]).rename(['SMOD2020']);

// b1-b7 :  2016-2022

var Ourss_2016 = imageF1.select('b1').remap([0,1,2,3,4,5,6,7],[0,0,0,0,0,0,1,2]).rename(['Ourss2016']);
var Ourss_2020 = imageF1.select('b5').remap([0,1,2,3,4,5,6,7],[0,0,0,0,0,0,1,2]).rename(['Ourss2020']);

var Images_Concat = ee.Image.cat(img_smod2015,img_smod2020,Ourss_2016,Ourss_2020).unmask();


var func_imp = function(f){
  f= ee.Feature(f);
  var Country = LSIB.filterBounds(f.geometry());
  
  Country = ee.Algorithms.If(Country.size().gte(1), Country.first().get('country_na'), 'NA');
  
  var f2 = f.set('Country',Country);

  var f_collection = ee.FeatureCollection([f]);
  
  var DHS_org = Images_Concat.sampleRegions({collection :f_collection, scale :10 }).first();
  
  f2 = f2.set('DHS_Ours2016',DHS_org.get('Ourss2016'),'DHS_Ours2020',DHS_org.get('Ourss2020'),
              'DHS_SMOD2015',DHS_org.get('SMOD2015'),'DHS_SMOD2020',DHS_org.get('SMOD2020'));
  
  for (var i=0; i<=19 ; i++){

    var propertyLon = ee.String('LonIm_').cat(ee.Number(i).format());
    var propertyLat = ee.String('LatIm_').cat(ee.Number(i).format());
    
    var Long = ee.Number.parse(f2.get(propertyLon));
    var Lat = ee.Number.parse(f2.get(propertyLat));
    
    var f_geo2 = ee.Geometry.Point([Long,Lat]);
    f2 = ee.Feature(f2.setGeometry(f_geo2))
    
    var DHS_imp = Images_Concat.sampleRegions({collection :ee.FeatureCollection([f2]), scale :10 }).first();
    
    f2 = f2.set('Imp_Ours2016_'+i.toString(),DHS_imp.get('Ourss2016'),'Imp_Ours2020_'+i.toString(),DHS_imp.get('Ourss2020'),
              'Imp_SMOD2015_'+i.toString(),DHS_imp.get('SMOD2015'),'Imp_SMOD2020_'+i.toString(),DHS_imp.get('SMOD2020'));
    
  }
  
  return f2;
};


var Rural_Urban_hsl_confusion = Rural_Urban_hsl.map(func_imp);

Export.table.toAsset(Rural_Urban_hsl_confusion, 'Rural_Urban_hsl_confusion', 'Africa/Rural_Urban_hsl_confusion');

