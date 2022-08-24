

var raster = new ol.layer.Tile({
	source: new ol.source.OSM()
});


var source = new ol.source.Vector({wrapX: false});

var vector = new ol.layer.Vector({
	source: source,
});




var map = new ol.Map({
    target: 'map',
    layers: [
		raster,
		vector
    ],
    view: new ol.View({
        center: ol.proj.fromLonLat([1.48, 43.55]),
        zoom: 12,
    }),



});


var geocoder = new Geocoder('nominatim', {
	provider: 'osm',
	lang: 'fr',
	placeholder: 'Chercher...',
	targetType: 'text-input',
	autoComplete: true,
	limit: 5,
	preventDefault: false
});

map.addControl(geocoder);

var typeSelect = document.getElementById('type')

let draw;


document.getElementById('polygone').addEventListener('click', function(){
	var nbr_features = vector.getSource().getFeatures().length;
	if(nbr_features == 0){
		draw = new ol.interaction.Draw({
			source: source,
			type: 'Polygon',
		});

		map.addInteraction(draw);
	}
});


document.getElementById('none').addEventListener('click', function(){
	map.removeInteraction(draw);
});



document.getElementById('generate').addEventListener('click', function(){
	var geom = []
	vector.getSource().forEachFeature(function(feature){geom.push(new ol.Feature(feature.getGeometry().clone().transform('EPSG:3857', 'EPSG:4326')));});
	var featureCount = geom.length;
	if (featureCount == 1){
		var mail = window.prompt("Enter your mail: ")
		var writer = new ol.format.GeoJSON();
		var geoJsonStr = writer.writeFeatures(geom);
		//sender = JSON.stringify(geoJsonStr, mail)

		//console.log(geoJsonStr);
		$.ajax({
			data : {'coords': geoJsonStr, 'mail': mail},
			url: "/square",
			type: 'POST',
			})
			.done(function(data){
				$('#output').text(data.output).show();
			});

	} else if (featureCount == 0){
		alert("Vous n'avez selectionner aucune emprise")
	} else{
		alert("Vous ne devez selectionner qu'une seule emprise")
	}

	/*
	emprise = JSON.stringify(geoJsonStr);
	const a = document.createElement("a");

	const file = new Blob([emprise], {type: "text/plain"});
	a.href = URL.createObjectURL(file);
	a.download = "emprise.json";
	a.click();
	*/

});

document.getElementById('undo').addEventListener('click', function(){
	var features = vector.getSource().getFeatures();
	features.forEach((feature) => {
		vector.getSource().removeFeature(feature);
	});
});


