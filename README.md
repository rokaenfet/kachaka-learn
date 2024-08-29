# kachaka-learn

Learning kachaka-api

# Execute
1. `venv/Source/activate`
2. `pip install -r requirements.txt`

# Debug / Profile
`scalene --cpu --memory --cli .\stalk\main.py`

# Command List
![synchronous API calls](https://github.com/pf-robotics/kachaka-api/blob/main/docs/kachaka_api_client.ipynb)
![asynchronous API calls](https://github.com/pf-robotics/kachaka-api/blob/main/docs/kachaka_api_client_async.ipynb)

# TODO
- [x] learn kachaka
- [x] teleop with live cam and lidar display
  - cam is laggy even tho fps is high
- [x] make kachaka move towards a person it sees
- [x] add face detection pipeline and take a snap of the front side w face in frame