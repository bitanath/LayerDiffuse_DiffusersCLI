from flask import Flask,request,make_response

app = Flask(__name__)

print("Warming up:")

@app.route('/ping', methods=['GET'])
def pinging():
    print("Got ping")
    return 'pong'

@app.route('/', methods=['GET'])
def index():
    print("Got index")
    return 'index'

@app.route('/metadata', methods=['POST'])
def json_example():
    request_data = request.get_json() #TODO get the base64 encoded thumbnail image of the design here
    request_headers = request.headers #TODO check the headers for requisite identification

    #return a dict to return a JSON by default
    response = make_response({
        "test": "this"
    })
    response.headers.set("Content-Type","application/json")
    return response



app.run(debug=True, port=8080, use_reloader=False)