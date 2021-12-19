import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__, template_folder='template')
logRegClf = pickle.load(open('LogRegModel.pkl', 'rb'))
svmRegClf = pickle.load(open('SVMModel.pkl', 'rb'))
treeRegClf = pickle.load(open('TreeModel.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predictLog():
    '''
    For rendering results on HTML GUI
    '''

    # Logistic Regression Prediction
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    predictionLog = logRegClf.predict(final_features)
    if predictionLog == 0:
        safeLog = 'not safe'
    else:
        safeLog = 'safe'

    # SVM Prediction
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    predictionSVM = svmRegClf.predict(final_features)
    if predictionSVM == 0:
        safeSVM = 'not safe'
    else:
        safeSVM = 'safe'

    # Decision Tree Prediction
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    predictionTree = treeRegClf.predict(final_features)
    if predictionTree == 0:
        safeTree = 'not safe'
    else:
        safeTree = 'safe'

    message = "Log Reg: Water is: " + safeLog + "\nSVM: Water is: " + \
        safeSVM + "\nDec. Tree: Water is: " + safeTree + "\n"

    return render_template('index.html', prediction_text=message)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
