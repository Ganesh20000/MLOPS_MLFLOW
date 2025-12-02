import mlflow
print(mlflow.get_tracking_uri())
print("this is before the url")

print("\n")


mlflow.set_tracking_uri('http://127.0.0.1:5000')
print("after the url tracking")

