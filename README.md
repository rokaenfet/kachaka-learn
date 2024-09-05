# kachaka-learn

Learning kachaka-api
think-pad pass:1234

# Execute
1. activate venv
2. `pip install -r requirements.txt; pip install -r ./MEBOW/requirements.txt`

# Folder
- `asset`
  - data
- `cocoapi`
- `demo`
  - test files for learning to use kachakaAPI
- `dev`
  - files produced during the stage of development
- `lib`
  - custom libraries. Includes libs from other repos
- `MEBOW`
  - cloned repo for HOE
- `openpose`
  - cloned repo for HOE (Pose detection)
- `stalk`
  - contains files used during the make of main program

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
- [x] asynchronous control of multiple robots
- [x] navigation system for multiple robots
  - set points to go to, or be able to share coordinates between both robots
- [x] camera / video setup which displays more continuosly
- [x] smooth switching between navigate and human_detection
- [ ] make kachaka move to the front side of the person via HOE or pose detection
  - [x] MP pose landmark > orientation for *full body*
- [ ] move kachaka and external devices with mounted laptop
  - [x] kachaka
  - [ ] robot arm
    - [ ] DAMIAO motor
  - [ ] camera
- [ ] Test model speeds
  - [ ] human detection
  - [ ] face detection
  - [ ] HOE


# Concurrency

![](asset/Python-Concurrency-API-Choice.png)

![](asset/Python-Concurrency-API-Decision-Tree.png)

[resource](https://superfastpython.com/python-concurrency-choose-api/#Problem_of_Pythons_Concurrency_APIs)

1. Choosing a module
   1. Coroutine-based using `asyncio`
   2. Thread-based using `threading`
   3. Process-based using `multiprocessing`
2. pool-based (e.g. process Pool) or class-based (e.g. Process class)?
   1. If using pool-based, Pool class or PoolExecutor class?

# Kachaka
- [specs](https://kachaka.life/technology/)

# Human-body orientation estimation

![](asset/demo.gif)

![](asset/demo1.JPG)

- **Monocular Estimation of Body Orientation in the Wild**
  - [repo](https://github.com/ChenyanWu/MEBOW)
  - Works well camera captures whole body, except when facing away (angle prediction oscillates aggressively), otherwise unusable
- **Partial-Human Orientation Estimation**
  - [repo](https://github.com/zhaojieting/Part_HOE)
  - [paper](https://arxiv.org/pdf/2404.14139)
  - No pre-trained model, in contact with author
- **OpenPose**
  - [repo](https://github.com/CMU-Perceptual-Computing-Lab/openpose/)
  - [pytorch implementation repo](https://github.com/Hzzone/pytorch-openpose?tab=readme-ov-file)
  - too slow (suggested usage w gpu)
- **MP Pose**
  - [web](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python)
  - only works with full body