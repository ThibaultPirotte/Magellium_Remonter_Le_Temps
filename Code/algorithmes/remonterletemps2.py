import shapefile
from shapely.geometry import shape, Point, Polygon, mapping
from shapely import wkt
import json
from osgeo import gdal, ogr
from autogeoreferencing2 import *
from multiprocessing.pool import Pool as ThreadPool
import time
import sys
import os
import cv2 as cv

tic = time.time()
download_time = 0

gdal.AllRegister()
current_dir = '/home/thpiro/RemonterLeTemps/FrontEnd'
os.chdir(current_dir)

#Choix de l'année et de l'emprisec
#emprise = "../Test/emprise5.shp"

if len(sys.argv) == 4 :
    first_year = min(int(sys.argv[2]), int(sys.argv[3]))
    end_year = max(int(sys.argv[2]), int(sys.argv[3]))

    if first_year < 1919 or first_year > 2010 or end_year < 1919 or end_year > 2010:
        print("Erreur: La fourchette temporelle doit être situé entre 1919 et 2010")
        quit()

elif len(sys.argv) == 2:
    first_year = 1919
    end_year = 2010

else:
    print("Nombre d'arguments incorrect. Doit être donné l'emprise et optionellement la fourchette temporelle")
    quit()


emprise = sys.argv[1]

if os.path.exists(emprise) == False:
    print("Le fichier n'existe pas")
    quit()

ext = os.path.splitext(emprise)[1]

if ext == '.shp':
    r = shapefile.Reader(emprise)
    shapes = r.shapes()
    poly1 = shape(shapes[0])

elif ext == ".geojson":
    with open(emprise,"r") as r: data = json.load(r)
    poly1 = Polygon([tuple(l) for l in data['features'][0]['geometry']['coordinates'][0]])
    print(poly1)
else:
    print("Le fichier passé en argument doit être un fichier .shp ou .geojson")
    quit()




chrono = np.flip(np.arange(first_year,end_year+1))
distance_btw_years = 1

#r = shapefile.Reader(emprise)
#Pas de données pour 1928 et 1941
chrono = np.delete(chrono, np.where(chrono == 1928))
chrono = np.delete(chrono, np.where(chrono == 1941))

important_year = []
last_year = 2019


print(poly1)

#shapes = r.shapes()
#poly1 = shape(shapes[0])

for year in chrono:
    with open("../../MetaDonnées/pva_"+str(year)+".geojson","r") as f: data = json.load(f)
    i = 0
    print(year)

    if data['features'] != [] and last_year - year > distance_btw_years:
        years = []
        im_ids = []
        names = []
        surfaces = []

        intersection = 0

        for elem in data['features']:
            lon = elem['properties']['lon']
            lat = elem['properties']['lat']
            center = Point(lon, lat)

            coin1 = Point(elem['geometry']['coordinates'][0][0][0], elem['geometry']['coordinates'][0][0][1])
            coin2 = Point(elem['geometry']['coordinates'][0][2][0], elem['geometry']['coordinates'][0][2][1])
            coin3 = Point(elem['geometry']['coordinates'][0][4][0], elem['geometry']['coordinates'][0][4][1])
            coin4 = Point(elem['geometry']['coordinates'][0][6][0], elem['geometry']['coordinates'][0][6][1])

            if poly1.contains(center) or poly1.contains(coin1) or poly1.contains(coin2) or poly1.contains(coin3) or poly1.contains(coin4):
                name = "Image_"+str(i)
                im_id = elem['properties']['idcliche']

                coinlist = [coin1, coin2, coin3, coin4]

                poly2 = Polygon([[p.x, p.y] for p in coinlist])
                inter = poly1.intersection(poly2)

                surface = inter.area/poly1.area

                if elem['properties']['url'] != None and inter.area/poly1.area > 0.2 and elem['properties']['type'] != 'IR' and elem['properties']['type']!='IRC':
                    intersection = intersection + inter.area/poly1.area
                    years.append(str(year))
                    im_ids.append(im_id)
                    names.append(name)
                    surfaces.append(elem)
                    i = i+1
    print(i) 
    n = i
    if i>8:
        i = 8

    if i > 0 and intersection > 0.75:
        toc = time.time()
        try:
            ThreadPool(i).starmap(img_finder, zip(years,im_ids,names))
        except ValueError:
            print("Erreur dans le telechargement des images")
        download_time += time.time() - toc
        image_name = "../stockage2/Image_"

        if n > 1:
            georeferenced = []
            no_georeferenced = list(range(0,n))
            junk_image = []

            distance = np.zeros(n)
            n = 0
            for elem in surfaces:
                distance[n] = (poly1.centroid.coords[0][0] - elem['properties']['lon'])**2 + (poly1.centroid.coords[0][1] - elem['properties']['lat'])**2
                n = n+1

            georeferenced.append(np.argmin(distance))
            no_georeferenced.pop(np.argmin(distance))

            print(im_ids[np.argmin(distance)])

            distance_between_image = np.ones([n,n], dtype = float)
            n = 0
            for elem1 in surfaces:
                m = 0
                for elem2 in surfaces:
                    distance_between_image[n][m] = (elem1['properties']['lon'] - elem2['properties']['lon'])**2 + (elem1['properties']['lat'] - elem2['properties']['lat'])**2
                    m = m+1
                n = n+1

            distance_between_image[distance_between_image == 0] = 999

            idx = np.argmin(distance_between_image[georeferenced, no_georeferenced])


            dst2 = image_name + str(georeferenced[0]) + ".tif"
            dst1 = image_name + str(no_georeferenced[idx]) + ".tif"

            err = georeferencing2(dst1, dst2, 20)
            if err == False:
                georeferenced.append(no_georeferenced[idx])
            else:
                junk_image.append(no_georeferenced[idx])
            no_georeferenced.pop(idx)
            georeferenced.sort()



        elif n == 1:
            no_georeferenced = []
            georeferenced = [0]

        while no_georeferenced != []:
            idx = np.argwhere(distance_between_image[georeferenced][:,no_georeferenced] == np.min(distance_between_image[georeferenced][:,no_georeferenced]))
            dst2 = image_name + str(georeferenced[idx[0][0]]) + ".tif"
            dst1 = image_name + str(no_georeferenced[idx[0][1]]) + ".tif"

            err = georeferencing2(dst1, dst2,20)
            if err == False:
                georeferenced.append(no_georeferenced[idx[0][1]])
                georeferenced.sort()
            else:
                junk_image.append(no_georeferenced[idx[0][1]])
            
            no_georeferenced.pop(idx[0][1])

        images_in_mosaic = []
        dst = "../static/IMG/" + str(year) + ".tif"
        for i in range(n):
            if i in georeferenced:
                img = "../stockage2/Image_"+str(i)+".tif"
                images_in_mosaic.append(img)
        
        Outfile = gdal.Warp(dst, images_in_mosaic, format = "GTiff")
        Outfile = None

        if important_year != []:
            img = cv.imread(dst,0)
            #img = reduce_size(img, 10000)
            #cv.imwrite(dst, img)
            year_reference = important_year[-1]
            dst2 = "../static/IMG/" + str(year_reference) + ".tif"

            err = georeferencing2(dst, dst2, 10)
            
            if err == True and len(important_year) >= 2:
                year_reference = important_year[-2]
                dst2 = "../static/IMG/" + str(year_reference) + ".tif"
                err = georeferencing2(dst, dst2, 10)

            if err == True:
                Outfile = gdal.Warp(dst, images_in_mosaic, format = "GTiff")
                Outfile = None

        Outfile = gdal.Warp(dst, dst, cutlineDSName = emprise, cropToCutline = True, dstNodata = 0)
        Outfile = None
        im = cv.imread(dst,0)
        w,h = im.shape
        hist = cv.calcHist([im], [0], None, [256], [0,256] )/(w*h)
        print(hist[0][0])
        if hist[0][0] < 0.4:
            important_year.append(year)
            last_year = year
        else:
            os.remove(dst)



convert_tif_to_png()

print(important_year)
print(time.time() - tic)
print(download_time)