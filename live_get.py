import requests

request = requests.get(url="https://predicting-salary.onrender.com")

print(request.status_code)
print(request.text)
