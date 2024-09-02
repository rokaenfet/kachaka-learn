import kachaka_api

KACHAKA_HOST = "192.168.118.158"
# grpc port
KACHAKA_PORT = 26400

client = kachaka_api.KachakaApiClient("192.168.118.158:26400")
error_code = client.get_robot_error_code()
print(error_code)