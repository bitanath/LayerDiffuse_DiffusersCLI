# GET requests will be blocked
from flask import Flask,request,make_response

app = Flask(__name__)

@app.route('/ping', methods=['GET'])
def pinging():
    return 'pong'

@app.route('/json-example', methods=['POST'])
def json_example():
    request_data = request.get_json()

    print(request_data)

    language = None
    framework = None
    python_version = None
    example = None
    boolean_test = None

    if request_data:
        if 'language' in request_data:
            language = request_data['language']

        if 'framework' in request_data:
            framework = request_data['framework']

        if 'version_info' in request_data:
            if 'python' in request_data['version_info']:
                python_version = request_data['version_info']['python']

        if 'examples' in request_data:
            if (type(request_data['examples']) == list) and (len(request_data['examples']) > 0):
                example = request_data['examples'][0]

        if 'boolean_test' in request_data:
            boolean_test = request_data['boolean_test']
    #return a dict to return a JSON by default
    response = make_response({
        "test": "this"
    })
    response.headers.set("Content-Type","application/json")
    return response


if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)