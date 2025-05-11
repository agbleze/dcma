
from api.helpers import request_prediction

in_data = {'category_id': 2, 'customer_id': '1oIA', 
           'publisher': 'LinkedIn',
            'industry': 'Tech', 
            'market_id': '21213'
        }
#URL = "http://0.0.0.0:5000/predict"
URL = "http://127.0.0.1/predict"
URL = "http://127.0.0.1:8080/predict"
def test_request_prediction():
    result = request_prediction(data=in_data, URL=URL)
    print(f"API CPA prediction: {result}")
    assert isinstance(result, float)
    
test_request_prediction()


