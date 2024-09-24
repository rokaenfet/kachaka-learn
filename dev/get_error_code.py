import kachaka_api

KACHAKA_HOST = "192.168.118.168:26400"
# grpc port
KACHAKA_PORT = 26400

client = kachaka_api.KachakaApiClient(KACHAKA_HOST)
error_code = client.get_robot_error_code()
print(error_code[10001])