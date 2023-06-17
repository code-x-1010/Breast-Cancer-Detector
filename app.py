from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
import webbrowser
from tensorflow.keras.utils import load_img
import pickle

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

info = {}

print("+"*50, "loadin gmmodel")
model = load_model('./mammography_pred_model.h5')


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict', methods=["POST"])
def prediction():
	info['lat'] = request.form['lat']
	info['view'] = request.form['view']
	info['age'] = request.form['age']
	info['bio'] = request.form['bio']
	info['inv'] = request.form['inv']
	info['imp'] = request.form['imp']
	info['den'] = request.form['den']
	img = request.files['img']
	img = img.filename
	img = load_img('./rsna-breast-cancer-512-pngs/'+img)
	# print(type(img))
	# print(img)
	arr = np.asarray(img)
	print(arr.shape)
	arr = arr.reshape(1, arr.shape[0], arr.shape[1], arr.shape[2]) / 255
	info['path'] = model.predict(arr)[0][0]
	print(info)
	density = {'B': 2, 'A': 1, 'C': 3, 'D': 4, np.nan: 0}
	info['den'] = density[info['den']]
	lat = {'L': 0, 'R': 1}
	view = {'CC': 0, 'MLO': 1, 'ML': 2, 'AT': 3}
	info['lat'] = lat[info['lat']]
	info['view'] = view[info['view']]
	print(info)
	arr = []
	for i in info.values():
		arr.append(float(i))
	X = np.array(arr).reshape((1,-1))
	print(X)
	classifier = pickle.load(open('decision_model.sav', 'rb'))
	pred = classifier.predict(X)



	return render_template("predict.html", data=pred)

if __name__ == "__main__":
	app.run(debug=True)