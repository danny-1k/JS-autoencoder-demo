from flask import Flask,render_template,request

import numpy as np

import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
@app.route('/generate')
def generate():
    return render_template('index.html',is_here=1)


@app.route('/interpolate')
def interpolate():
    return render_template('interpolate.html',is_here=2)


@app.route('/get/w/<sub>/<idx>')
def get_weights(sub,idx):
    f =  open(f'static/{sub}/w{idx}.txt').read()
    return f

@app.route('/get/b/<sub>/<idx>')
def get_biases(sub,idx):
    f =  open(f'static/{sub}/b{idx}.txt').read()
    return f


@app.route('/get_inter/<which>')
def get_inter(which):
    
    which = str(which)

    if which in ['ankle_boot','pullover','sandal','sneaker','trouser']:
        f = open(f'static/{which}.txt').read()
    
    else:
        f = open('static/ankle_boot.txt').read()

    return f
    
if __name__ == '__main__':
    app.run(debug=True)