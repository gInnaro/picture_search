from flask import *
from pixellib.torchbackend.instance import instanceSegmentation
from PIL import Image
from dicts import libs
import shutil

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def main():
    if request.method == 'POST':
        dist = 'static/image.jpg'
        answer = []
        answer_dict = {}
        f = request.files['file']
        file = f.filename
        shutil.copy2(file, dist)
        im = Image.open(file)
        w, h = im.size
        ins = instanceSegmentation()
        ins.load_model("pointrend_resnet50.pkl")
        text = ins.segmentImage(file)
        for txt in text[0]['class_names']:
            answer.append(dictationary[txt])
        for i in range(len(answer)):
            if answer_dict.get(answer[i]) == None:
                answer_dict[answer[i]]=1
            else:
                key = answer_dict[answer[i]] + 1
                answer_dict[answer[i]] = key
        return render_template("Acknowledgement.html", width=w, height=h, dicts=answer_dict)
    else:
        return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)


dictationary = dict(libs())





