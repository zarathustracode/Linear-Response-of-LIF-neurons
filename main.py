
from fastapi import FastAPI
from SynapticCurrent import Current, NeuronalResponse
from ioLIF import *

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



