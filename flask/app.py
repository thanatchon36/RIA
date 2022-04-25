from flask import Flask, send_from_directory
import os

app = Flask(__name__)

workingdir = os.path.abspath(os.getcwd())
workingdir = workingdir + '/RIA_Input/'

@app.route("/pdf/<path1>/<path2>/<path3>/<path4>")
def pdf(path1,path2,path3,path4):
    file_path = workingdir + "{}/{}/{}/".format(path1, path2, path3)
    return send_from_directory(file_path, path4)

if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = 8601)
    
# http://pc140032646.bot.or.th:8601/pdf/Clean_International_Regulator_OCR/25-MAS/other/sg_56.pdf