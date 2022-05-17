
from fastapi import FastAPI, Path, Query
from .SynapticCurrent import Current, NeuronalResponse
from .ioLIF import *

app = FastAPI(
        title="Leaky integrate-and-fire neurons API",
        description="API endpoint for Python/C++ program")

@app.get('/')
async def index():
    return {'message': 'This is LIF API'}


@app.post('/neuron')
async def membrane_potential(data: Current):
    data = data.dict()
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
        mean_val: float = Query(default=0., gt=-10, le=10),
        variance_val:float = Query(default=1., ge=1, le=5),
        tag: str = Path(default='calculate')):

    V,P0,J0,r0 = LIF0(tau=tau_mem,mu=mean_val,sig=variance_val,Vth=V_th, Vresting=V_resting,Vre=V_reset,tau_ref=t_ref)

    return {
            'membrane potential values': list(V),
            'probability distribution': list(P0),
            'probability flux': list(J0),
            'tag': tag
            }
