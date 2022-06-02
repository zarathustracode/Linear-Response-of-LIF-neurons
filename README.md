# Linear-Response-of-LIF-neurons

Linear response approach to solving spiking-neural PDE with a fastAPI endpoint. 

# Run

To start the app run

`uvicorn lifAPI.main:app --host=0.0.0.0 --reload`

# Docker

To create docker container run

`docker build -t leaky-integrate-and-fire:0.1 .`

 and to run it

`docker run -p 8000:8000 --name lif-api leaky-integrate-and-fire:0.1`
