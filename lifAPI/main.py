
from fastapi import FastAPI, Path, Query, Request
from .SynapticCurrent import Current
from .ioLIF import *
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder

app = FastAPI(
        title="Leaky integrate-and-fire neurons API",
        description="API endpoint for Python/C++ program")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
async def index(request: Request):
    print(request.headers)
    return {'message': 'This is LIF API'}


@app.post('/neuron')
async def membrane_potential(data: Current):
    #data = data.dict()
    data = jsonable_encoder(data)
    mean = data['mean']
    variance = data['variance']

    V,P0,J0,r0 = LIF0(tau=tau_mem,mu=mean,sig=variance,Vth=V_th, Vresting=V_resting,Vre=V_reset,tau_ref=t_ref)

    return {
            'membrane potential values': list(V),
            'probability distribution': list(P0),
            'probability flux': list(J0)
            }

@app.get("/get_stationary_diststribution/{tag}/")
async def get_membrane_potential(
        mean_val: float = Query(default=20., gt=-2, le=30),
        variance_val:float = Query(default=5., ge=1, le=10),
        tag: str = Path(default='calculate')):

    V,P0,J0,r0 = LIF0(tau=tau_mem,mu=mean_val,sig=variance_val,Vth=V_th, Vresting=V_resting,Vre=V_reset,tau_ref=t_ref)

    return {
            'grid': list(V),
            'probabilities': list(P0),
            'fluxes': list(J0),
            'tag': tag
            }
