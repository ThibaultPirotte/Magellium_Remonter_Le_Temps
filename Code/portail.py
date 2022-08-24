
from flask import Flask,render_template, request, jsonify
from flask_mail import Mail, Message
from shapely.geometry import shape, Point, Polygon, mapping
from pyproj import Geod
import json
import os
import glob

app = Flask(__name__)
IMG_FOLDER = os.path.join('static', 'IMG')
app.config['UPLOAD_FOLDER'] = IMG_FOLDER


#app.config['MAIL_SERVER'] = 'smtp.live.com'
#app.config['MAIL_PORT'] = 587
#app.config['MAIL_USERNAME'] = 'thibault.pirotte@live.com'
#app.config['MAIL_PASSWORD'] = 'Go2thesun.'
#app.config['MAIL_USE_TLS'] = True
#app.config['MAIL_USE_SSL'] = False
#mail = Mail(app)


@app.route("/", methods=['GET'])
def test():
    return render_template('portail.html')

@app.route("/square", methods = ['POST'])
def square():
    dir = 'static/IMG'
    for f in os.listdir(dir):
      os.remove(os.path.join(dir, f))


    dir = 'stockage2'
    for f in os.listdir(dir):
      os.remove(os.path.join(dir, f))


    #coord = request.get_data()
    coord = request.values.get("coords")
    mail = request.values.get("mail")
    print(coord)
    print(mail)
    #coord = coord.decode("utf-8")
    coord2 = json.loads(coord)
    with open('data.geojson','w') as f:
      json.dump(coord2, f)
    coord2 = coord2['features'][0]['geometry']['coordinates'][0]
    poly = Polygon(coord2)


    geod = Geod(ellps="WGS84")
    area = abs(geod.geometry_area_perimeter(poly)[0]) /1000000
    print('#area: {:.3f} km^2'.format(area))
    os.system("python3 algorithmes/remonterletemps2.py data.geojson")




    #msg = Message(
    #  'Hello',
    #  sender = 'thibault.pirotte@live.com',
    #  recipients = ['thibault.pirotte@magellium.fr']
    #  )
    #msg.body = "This is a test"
    #print(msg)
    #mail.send(msg)
    return ("hello")


@app.route("/results")
def Display_IMG():
  IMG_LIST = os.listdir('static/IMG')
  IMG_LIST = ['IMG/' + i for i in IMG_LIST]
  IMG_LIST.sort(reverse = True)
  years = [x.replace('IMG/','').replace('.jpg','') for x in IMG_LIST]
  #years.sort(reverse = True)
  nbr_IMG = len(IMG_LIST)
  return render_template('resultats.html', imagelist = IMG_LIST, nbr_img = nbr_IMG, years_image = years) 


if __name__ == "__main__":
  app.run()