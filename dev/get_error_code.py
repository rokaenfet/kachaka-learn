import kachaka_api

KACHAKA_HOST = "192.168.118.158"
# grpc port
KACHAKA_PORT = 26400

client = kachaka_api.KachakaApiClient()
error_code = client.get_robot_error_code()
print(error_code)