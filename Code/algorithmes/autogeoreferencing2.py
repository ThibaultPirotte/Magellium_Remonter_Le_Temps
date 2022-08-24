import numpy as np
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform
import cv2 as cv
from osgeo import gdal, osr
import json
import os
from os import listdir
import requests
import numpy as np
import time
from rootsift import RootSIFT

#Toolbox contenant toutes les fonctions nécéssaires au géoreferencement automatique des images



#Permet de sélectionner des points de calages potentiels
#entre 2 images à l'aide de l'algorithme SIFT
def kp_selector(img1,img2):
	#On met les images en niveaux de gris pour être dans le même spectre de couleur
	#tic = time.time()
	#img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
	#img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
	
	img1, n1 = reduce_size(img1, 5000)
	img2, n2 = reduce_size(img2, 5000)

	#Pour chaque image, on detecte les descripteurs
	detector = cv.SIFT_create()

	kp1 = detector.detect(img1)
	kp2 = detector.detect(img2)

	rs = RootSIFT()

	(kp1, desc1) = rs.compute(img1, kp1)
	(kp2, desc2) = rs.compute(img2, kp2)

	#Pour chaque descripteurs de l'image 1, on cherche les 2 plus proche voisin dans les
	#descripteurs de l'image 2.
	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

	search_params = dict(checks = 50)
	flann = cv.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(desc1, desc2, k=2)

	#On ne garde que les matches où la distance par rapport au plus proche voisin
	#est nettement inférieur par rapport à la distance aux 2eme plus proche voisin.
	good = []
	matches = filter(None, matches)
	for m, n in matches:
		if m.distance < 0.7*n.distance:
			good.append(m)
	
	if(len(good)>1000):
		np.random.shuffle(good)
		good = good[0:1000]

	#Pour chaque match on prend les coordonnées des descripteurs des 2 images
	src_pts = np.float32([kp1[m.queryIdx].pt for m in good ]).reshape(-1,2)
	dst_pts = np.float32([kp2[m.trainIdx].pt for m in good ]).reshape(-1,2)
	
	#RANSAC
	#Afin d'éliminer les potentielles mauvais résultats possible, on utilise l'algorithme
	#du RANSAC. Les bons résultats sont stockés dans inliers.
	if len(good)>4:
		model, inliers = ransac(
			(src_pts, dst_pts),
			AffineTransform, min_samples = 4,
			residual_threshold = 20, max_trials = 10000
		)

	else:
		coord_in = [None]
		coord_out = [None]
		return coord_in, coord_out

	n_inliers = np.sum(inliers)
	print(n_inliers)

	inlier_keypoints_left = [cv.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
	inlier_keypoints_right = [cv.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
	placeholder_matches = [cv.DMatch(idx, idx, 1) for idx in range(n_inliers)]
	coord_in = [inlier_keypoints_left[match.queryIdx].pt for match in placeholder_matches]
	coord_out = [inlier_keypoints_right[match.trainIdx].pt for match in placeholder_matches]

	

	#Code pour visualiser les correspondances
	#img3 = cv.drawMatches(img1, inlier_keypoints_left, img2, inlier_keypoints_right, placeholder_matches, None)
	#print(time.time() - tic)
	#cv.namedWindow("output", cv.WINDOW_NORMAL)
	#cv.imshow("output", img3)
	#cv.waitKey(0)
	#cv.destroyAllWindows()



	coord_in = [(coord[0]*n1,coord[1]*n1) for coord in coord_in]
	coord_out = [(coord[0]*n2,coord[1]*n2) for coord in coord_out]



	return coord_in, coord_out

#Géoréférence l'image contenu dans dst1 par l'image contenue dans dst2
def georeferencing2(dst1, dst2, n):
	img1 = cv.imread(dst1,0)
	img2 = cv.imread(dst2,0)


	nb_kp = 0
	try:
		coord_in, coord_out = kp_selector(img1, img2)
		nb_kp = len(coord_in)
	except Exception as e:
		print("Erreur detecté")
		return True
		
	print(nb_kp)

	err = False

	if nb_kp > n:    #limite arbitraire de nombre de points de calage pour qu'on considère que le géoreferencement est bon
		img1 = gdal.Open(dst1, gdal.GA_Update)
		img2 = gdal.Open(dst2)
		coord_out2 = coord_transform(img2, coord_out)
		sr = osr.SpatialReference()
		sr.SetWellKnownGeogCS('WGS84')
		gcps = []
		for i in range(nb_kp):
			gcp = gdal.GCP(coord_out2[i][0], coord_out2[i][1], 0, coord_in[i][0], coord_in[i][1])
			gcps.append(gcp)
		
		img1.SetGCPs(gcps,sr.ExportToWkt()) #Transformation en fonction des points choisis.
		img1.SetProjection(sr.ExportToWkt())
		img1.SetGeoTransform(gdal.GCPsToGeoTransform(gcps))
		print("Georeferencing done")
	
	else:
		print("Cannot georeference the image because too few keypoints")
		err = True
	img1 = None
	img2 = None

	return err
	

#Transforme un jeu de coordoonées dans la projection de l'image
def coord_transform(im, coords):

	new_coord = []
	GT = im.GetGeoTransform()


	for coord in coords:
		x = GT[0] + coord[0]*GT[1] + coord[1]*GT[2]
		y = GT[3] + coord[0]*GT[4] + coord[1]*GT[5]
		new_coord.append([x,y])

	return new_coord


#Réduit la taille de l'image (SIFT ne marche pas sur une image trop grosse)
def reduce_size(im, n):
    i = 1
    h, w = im.shape
    while(h > n or w > n):
        im = cv.resize(im, (int(w/2), int(h/2)))
        h, w = im.shape
        i = i*2
    return im,i



#Permet d'éliminer les bords de l'image tout en conservant la transformation géo
def image_crop(dst, src):
	im = gdal.Open(src)
	w = im.RasterXSize
	h = im.RasterYSize

	m = int(np.round(h/8))
	n = int(np.round(w/8))

	window = (n,m, w-2*n, h-2*m)
	gdal.Translate(dst, src, srcWin = window)


def image_equalization(dst, mean, std):
	im  = gdal.Open(dst)
	geotransform = im.GetGeoTransform()

	im = cv.imread(dst,0)
	im = im - np.round(np.mean(im))
	im = (std/np.std(im)) * im
	im = im + mean
	im = np.round(im)
	im = np.clip(im,0,255)
	im[im == 0] = 1
	im = im.astype('uint8')
	cv.imwrite(dst, im)

	im = gdal.Open(dst, gdal.GA_Update)

	sr = osr.SpatialReference()
	sr.SetWellKnownGeogCS('WGS84')

	im.SetProjection(sr.ExportToWkt())
	im.SetGeoTransform(geotransform) 
	im = None


def convert_tif_to_png():
	dir = 'Resultats'
	for f in os.listdir(dir):
		src = dir + '/' + f
		dst = src.replace(".tif", ".jpg")
		im = gdal.Open(src, gdal.GA_Update)
		gdal.Translate(dst, im)
		os.remove(src)









#Permet de télécharger et géoréférencer une image
#Il suffit de connaître l'année et l'identifiant de l'image.
#L'image téléchargé sera au format TIF
def img_finder(year, image_id, name):

		with open("../../MetaDonnées/pva_"+year+".geojson", "r") as f:
			data = json.load(f)

		i = 0

	#Recherche de l'index correspondant à l'id, cette index est stocké dans i
		for elem in data['features']:
			if elem['properties']['idcliche'] == image_id:
				break
			i += 1
		elem = data['features'][i]



	#On récupère les élements importants pour le géoréférençage;
	#c-a-d les coordonnées et l'orientation
		Coord = elem['geometry']['coordinates'][0]
		Coord = [Coord[0], Coord[2], Coord[4], Coord[6]]
		orientation = float(elem['properties']['orientation']) #Utile pour placer les points ensuite

	#On standardise la position des coordonnées
		Coord.sort(key = lambda x:x[1], reverse = True)
		CoordUp = Coord[0:2]
		CoordDown = Coord[2:4]
		CoordUp.sort(key = lambda x:x[0])
		CoordDown.sort(key = lambda x:x[0], reverse = True)
		Coord = CoordUp + CoordDown


	#Télechargement de l'image
		url = 'https://wxs.ign.fr/x7yv499pbcguxhhxh8syehwe/jp2/DEMAT.PVA/'+elem['properties']['mission']+'/'+elem['properties']['url']
		print(url)
		r = requests.get(url, allow_redirects = True)
		dst2 = "../stockage1/"+name+".jp2"
		open(dst2, 'wb').write(r.content)


	#On transforme l'image en fichier TIF et on supprime l'ancien fichier JP2
		dst = ("../stockage1/"+name+".tif")
		im = gdal.Open(dst2, gdal.GA_Update)
		gdal.Translate(dst, im)
		os.remove(dst2)

	#Reduction de la taille des images pour faciliter l'exécution de l'algorithme SIFT

		im = cv.imread(dst,0)
		#im = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
		#im = cv.equalizeHist(im)

		im, rze = reduce_size(im, 7500)

		#mean = 160
		#std = 80
	
		#im = im - np.round(np.mean(im))
		#im = (std/np.std(im)) * im
		#im = im + mean
		#im = np.round(im)
		#im = np.clip(im,0,255)
		#im = im.astype('uint8')

		cv.imwrite(dst, im)

		im = gdal.Open(dst, gdal.GA_Update)
		xmax = im.RasterXSize
		ymax = im.RasterYSize


		sr = osr.SpatialReference()
		sr.SetWellKnownGeogCS('WGS84') #Les coordonnées sont dans le format WGS84

	#Construction des points de calage, à l'aide des coordonnées des coins dans les métadata et de l'orientation
		if orientation < 45 or orientation >= 315:
			pc1 = gdal.GCP(Coord[0][0], Coord[0][1], 0, 0,0)
			pc2 = gdal.GCP(Coord[1][0], Coord[1][1], 0, xmax-1,0)
			pc3 = gdal.GCP(Coord[2][0], Coord[2][1], 0, xmax-1,ymax-1)
			pc4 = gdal.GCP(Coord[3][0], Coord[3][1], 0, 0,ymax - 1)

		elif orientation < 135:
			pc1 = gdal.GCP(Coord[0][0], Coord[0][1], 0, 0,ymax-1)
			pc2 = gdal.GCP(Coord[1][0], Coord[1][1], 0, 0,0)
			pc3 = gdal.GCP(Coord[2][0], Coord[2][1], 0, xmax-1,0)
			pc4 = gdal.GCP(Coord[3][0], Coord[3][1], 0, xmax-1,ymax-1)

		elif orientation < 225:
			pc1 = gdal.GCP(Coord[0][0], Coord[0][1], 0, xmax - 1,ymax-1)
			pc2 = gdal.GCP(Coord[1][0], Coord[1][1], 0, 0,ymax - 1)
			pc3 = gdal.GCP(Coord[2][0], Coord[2][1], 0, 0,0)
			pc4 = gdal.GCP(Coord[3][0], Coord[3][1], 0, xmax-1,0)

		elif orientation < 315:
			pc1 = gdal.GCP(Coord[0][0], Coord[0][1], 0, xmax-1,0)
			pc2 = gdal.GCP(Coord[1][0], Coord[1][1], 0, xmax-1,ymax-1 )
			pc3 = gdal.GCP(Coord[2][0], Coord[2][1], 0, 0, ymax-1)
			pc4 = gdal.GCP(Coord[3][0], Coord[3][1], 0, 0,)

		gcps = [pc1,pc2,pc3,pc4]


		im.SetGCPs(gcps,sr.ExportToWkt()) #Assignement des points de calage à l'image
		im.SetProjection(sr.ExportToWkt())
		im.SetGeoTransform(gdal.GCPsToGeoTransform(gcps)) #Calcul de la transformation affine a partir des points de calage
		im = None

	#On crop les bords de l'image
		image_crop("../stockage2/"+name+".tif", dst)
		os.remove(dst)

		image_equalization("../stockage2/"+name+".tif",115,60)



