import numpy as np
from numpy import exp, pi
from numpy import sqrt
from scipy.special import erfcx
from .LIF0 import lif1py as LIFc
from .LIF0 import lifResponsepy as LIFr
from .LIF0 import getSolutionpy as frt
from scipy.integrate import quad, odeint


N           = 12500                              # total number of neurons in the network
f           = 0.8                               # fraction of excitatory neurons
epsilon_exc = 0.1                            # connectivity of the excitatory neurons of the network
epsilon_inh = 0.1                               # connectivity of the inhibitory neurons of the network
g           = 5.                                 # relative inhibitory strength

t_ref       = 2.                                # refractory period (ms)
tau_mem     = 20.                               # membrane time constant (ms)
V_reset     = 10.                               # reset potential (mV)
V_resting   = 0.
V_th        = 20.                               # firing threshold (mV)
J           = 0.1                               # weight of excitatory recurrent connections (mV) - amplitude of PSP
J_ext       = J                                 # weight of external connections (mV) - amplitude of PSP

eta         = 1.5                               # ratio between external rate and external frequency needed for the mean input to reach threshold in absence of feedback
nu_th       = V_th/(J*epsilon_inh*N*f*tau_mem)  # external frequency needed for the mean input to reach threshold in absence of feedback (/ms)
nu_ex       = eta*nu_th                         # external rate per individual external unit (/ms)
p_rate      = nu_ex*epsilon_inh*N*f             # total external rate per neuron (/ms)
#p_rate = 15.0 #This is different !!!! 

def jVre(V, Vre):
    if(V>Vre):
        return 1.
    else:
        return 0.

def LIF(tau=20.,mu=10.,Vresting=-60.,sig=5.,Vth=-50.,Vre=-60., tau_ref=2., RETURNVERB=True):

    E0 = Vresting + mu
    
    # set up the lattice for the discretized integration
    dV=0.01
    Vl=-10. + Vresting           # the lower bound for the voltage range
    V=np.arange(Vl,Vth+dV,dV)
    
    j = lambda V: jVre(V, Vre)
    
    dpdv = lambda p,V: - 2./(sig**2)*(V-E0) * p - 2.*tau/(sig**2) * j(V)
    
    p0 = odeint(dpdv,0, V[::-1])[::-1]
    jv = np.vectorize(j)
    j0 = jv(V)
    
    r0=1./(tau_ref+dV*np.sum(p0))  # steady-state firing rate (in kHz)
    P0=r0*p0           # correctly normalised density and current
    J0=r0*j0

    if RETURNVERB:
        return V,P0,J0,r0
    else:
        return r0

def LIF0(tau=20.,mu=10.,Vresting=-60.,sig=5.,Vth=-50.,Vre=-60., RETURNVERB=True, tau_ref=2.):
    '''
    #                                     LIF0
    #  Steady-state rate and density for the Leaky Integrate-and-Fire model 
    #  using Threshold Integration. 

    #  The voltage obeys the equation:
    #  tau*dVdt = E0 - V + sig*sqrt(2*tau)*xi(t)

    #  where xi(t) is Gaussian white noise with <xi(t)xi(t')>=delta(t-t')
    #  such that integral(t->t+dt) xi(t)dt =sqrt(dt)*randn

    #  The threshold is at Vth with a reset at Vre.

    #  input:  expected units are: tau [ms], sig,E0,Vth and Vre all in [mV]
    #  (example values might be tau=20; sig=5; E0=-50; Vth=-50; Vre=-60;)

    #  output: vectors V [mV] range of voltage
    #                    P0 [1/mV] probability density
    #                    J0 [kHz] probability flux (a piece-wise constant)
    #             scalar r0 [kHz] steady state firing rate
    '''    

    E0 = Vresting + mu
    
    # set up the lattice for the discretized integration
    dV=0.01
    Vl=-10. + Vresting           # the lower bound for the voltage range
    V=np.arange(Vl,Vth+dV,dV)
    n=len(V);
    tol = 1e-6 # Edit this to change tolerance
    kre=np.where( np.abs(V - Vre ) < tol )[0][0]   # NB Vre must fall on a lattice point!

    # Will use the modified Euler method (Phys. Rev. E 2007, Eqs A1-A6)
    G=(V-E0)/(sig**2/2.)    # the only part specific to the Leaky IF
    A=np.exp(dV*G)
    B=(A-1.)/((sig**2/2.) *dV*G)
    B[G==0]=1./(sig**2/2.)

    # set up the vectors for the scaled current and probability density
    j0=np.zeros(n)
    p0=np.zeros(n)
    j0[n-1]=1.        # initial conditions at V=Vth. NB p0(n)=0 already.

    for k in np.arange(n-1, 0,-1):    # integrate backwards from threshold

        j0[k-1]=j0[k] - int(k==kre+1);
        p0[k-1]=p0[k]*A[k] + dV*B[k]*tau*j0[k];


    r0=1./(tau_ref+dV*np.sum(p0))  # steady-state firing rate (in kHz)
    P0=r0*p0           # correctly normalised density and current
    J0=r0*j0

    if RETURNVERB:
        return V,P0,J0,r0
    else:
        return r0
    


def moments(rate):
    mu = tau_mem * (J_ext*p_rate + J*epsilon_exc*rate*f*N - J*rate*(1.-f)*N*epsilon_inh*g)
    sig= sqrt(tau_mem * (J_ext**2*p_rate + J**2*epsilon_exc*rate*f*N + J**2*rate*(1.-f)*N*epsilon_inh*g**2))
    return mu, sig

def loop(tol=1e-6,nmax=50, method='Brunel'):
    n=1
    rate = 1./1000.
    err=1.
    mu=0.
    sig=0.
    nu=0.
    while err>tol and n<nmax:
        mu,sig = moments(rate)
        if method=='Brunel':
            nu = (t_ref+tau_mem*np.sqrt(np.pi)*quad(lambda u:erfcx(-u),(V_reset-mu)/sig,(V_th-mu)/sig)[0])**-1
        elif method=='Richardson':
            nu = LIF0(tau=tau_mem,mu=mu,sig=sig,Vth=V_th, Vresting=V_resting,Vre=V_reset,tau_ref=t_ref, RETURNVERB=False)
        else:
            print("No such method!")
        err = abs(nu-rate)
        rate = nu
        n = n+1
        
    return rate, mu, sig, n

def optimizeB(x):
    mu,sig = moments(x[2])
    rate = (t_ref+tau_mem*np.sqrt(np.pi)*quad(lambda u:erfcx(-u),(V_reset-mu)/sig,(V_th-mu)/sig)[0])**-1
    return  -x[0]+mu, -x[1]+sig, -x[2]+rate


def optimizeR(x):
    mu,sig = moments(x[2])
    rate = LIF0(tau=tau_mem,mu=mu,sig=sig,Vth=V_th, Vresting=V_resting,Vre=V_reset,tau_ref=t_ref, RETURNVERB=False)
    return  -x[0]+mu, -x[1]+sig, -x[2]+rate

def optimizeC(x):
    mu,sig = moments(x[2])
    rate = LIFc(mu,  sig,  V_resting, V_th, V_reset, tau_mem,  t_ref)[0]
    return  -x[0]+mu, -x[1]+sig, -x[2]+rate


def LIF1w(tau=20.,mu=18.,Vresting=-60.,sig=5.,Vth=-40.,Vre=-60.,tref=2.,E1=0.1,w=0.314,dw0=1e-05):
    '''
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                                    LIF1
    % Leaky Integrate-and-Fire model with modulated parameters 
    % using Threshold Integration. 
    %
    % The code is written for current modulation (E) but can be easily modified
    % for conductance or noise modulation.
    %
    % The voltage obeys the equation:
    % tau*dVdt = E0 + E1*cos(w*t) - V + sig*sqrt(2*tau)*xi(t)
    %
    % where xi(t) is Gaussian white noise with <xi(t)xi(t')>=delta(t-t')
    % such that integral(t->t+dt) xi(t)dt =sqrt(dt)*randn
    %
    % The threshold is at Vth with a reset at Vre.
    %
    % The firing rate response is calculated to first order
    %  r(t)=r0 + abs(r1)*cos(w*t+phase(r1))
    %
    % input:  expected units are: tau [ms], sig,E0,E1,Vth and Vre all in [mV]
    %         and fkHz is the modulated frequency f=w/2*pi in kHz
    %
    % output:  r0 [kHz] steady state firing rate
    %          r1 [kHz] rate response (complex number)
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''

    E0 = Vresting + mu

    # set up the lattice for the discretized integration
    dV=0.01
    Vl=-80.            # the lower bound for the voltage range
    V=np.arange(Vl,Vth+dV,dV)
    n=len(V);
    tol = 1e-6 # Edit this to change tolerance
    kre=np.where( np.abs(V - Vre ) < tol )[0][0]   # NB Vre must fall on a lattice point!

    # Will use the modified Euler method (Phys. Rev. E 2007, Eqs A1-A6)
    G=(V-E0)/(sig**2/2.)    # the only part specific to the Leaky IF
    A=np.exp(dV*G)
    B=(A-1.)/((sig**2/2.) *dV*G)
    B[G==0]=1./(sig**2/2.)
    
    # set up the vectors for the scaled current and probability density
    j0=np.zeros(n)
    p0=np.zeros(n)
    j0[n-1]=1.        # initial conditions at V=Vth. NB p0(n)=0 already.

    for k in np.arange(n-1, 0,-1):    # integrate backwards from threshold

        j0[k-1]=j0[k] - int(k==kre+1);
        p0[k-1]=p0[k]*A[k] + dV*B[k]*tau*j0[k];

    #############################################
    # first need the steady state
    #############################################


    r0=1./(dV*np.sum(p0))  # steady-state firing rate (in kHz)
    P0=r0*p0           # correctly normalised density and current
    J0=r0*j0



    ############################################
    # now the response to current (E) modulation
    ############################################

    #w=2*np.pi*fkHz

    jh1=np.zeros(n,dtype=complex)
    jh1[n-1]=1
    ph1=np.zeros(n,dtype=complex)

    jhE=np.zeros(n,dtype=complex)
    phE=np.zeros(n,dtype=complex)
    
    if(w!=0):

        for k in np.arange(n-1, 0,-1):    # integrate backwards from threshold

            jh1[k-1]=jh1[k] + dV*1j*w*ph1[k] - int(k==kre+1)
            ph1[k-1]=ph1[k]*A[k] + dV*B[k]*tau*jh1[k]

            # the 2nd equation in the following pair is specific to E modulation
            # if conductance or noise modulation is required then
            # it is only the last (inhomogeneous) term containing P0 that must be
            # altered - please see Eq 33 of Phys. Rev. E (2007) paper for details
            jhE[k-1]=jhE[k] + dV*1j*w*phE[k]
            phE[k-1]=phE[k]*A[k] + dV*B[k]*(tau*jhE[k] - E1*P0[k])


        r1=-jhE[0]/jh1[0]       # because Jh(Vl)=0 => jhe(1) + r1*jh1(1)=0
        
    else:
        r1=0.0j
    
#     rho = 0. * 1j
#     spikecov = 0. * 1j
    
#     if(w!=0):
#         rho = r1/(1.-r1) 
#     else:
#         rho=rho+np.pi*r0/dw0;          # this requires a dw to be correctly defined

#     spikecov=r0*(1.+2.*np.real(rho))

    return V,P0,J0,r0, r1 #, rho, spikecov

def GetTheFilter(dw, w, hx, t1=-100,t2=500,dt=0.5):
    nf = len(w)
    t = np.arange(t1, t2+dt, dt)
    nt = len(t)
    
    x = np.zeros(nt)+1j
    
    for i,s in enumerate(t):
        x[i] = (1./(2.*np.pi))*np.sum(dw*np.exp(1j*w*s)*hx)
        
    return t, x

def LIF2(tau=20.,mu=18.,Vresting=-60.,sig=5.,Vth=-40.,Vre=-60.,tref=2.,w=0.314,dw0=1e-05):
    '''
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                                    LIF1
    % Leaky Integrate-and-Fire model with modulated parameters 
    % using Threshold Integration. 
    %
    % The code is written for current modulation (E) but can be easily modified
    % for conductance or noise modulation.
    %
    % The voltage obeys the equation:
    % tau*dVdt = E0 + E1*cos(w*t) - V + sig*sqrt(2*tau)*xi(t)
    %
    % where xi(t) is Gaussian white noise with <xi(t)xi(t')>=delta(t-t')
    % such that integral(t->t+dt) xi(t)dt =sqrt(dt)*randn
    %
    % The threshold is at Vth with a reset at Vre.
    %
    % The firing rate response is calculated to first order
    %  r(t)=r0 + abs(r1)*cos(w*t+phase(r1))
    %
    % input:  expected units are: tau [ms], sig,E0,E1,Vth and Vre all in [mV]
    %         and fkHz is the modulated frequency f=w/2*pi in kHz
    %
    % output:  r0 [kHz] steady state firing rate
    %          r1 [kHz] rate response (complex number)
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''

    E0 = Vresting + mu

    # set up the lattice for the discretized integration
    dV=0.01
    Vl=-80.            # the lower bound for the voltage range
    V=np.arange(Vl,Vth+dV,dV)
    n=len(V);
    tol = 1e-6 # Edit this to change tolerance
    kre=np.where( np.abs(V - Vre ) < tol )[0][0]   # NB Vre must fall on a lattice point!

    # Will use the modified Euler method (Phys. Rev. E 2007, Eqs A1-A6)
    G=(V-E0)/(sig**2/2.)    # the only part specific to the Leaky IF
    A=np.exp(dV*G)
    B=(A-1.)/((sig**2/2.) *dV*G)
    B[G==0]=1./(sig**2/2.)
    
    # set up the vectors for the scaled current and probability density
    j0=np.zeros(n)
    p0=np.zeros(n)
    j0[n-1]=1.        # initial conditions at V=Vth. NB p0(n)=0 already.

    for k in np.arange(n-1, 0,-1):    # integrate backwards from threshold

        j0[k-1]=j0[k] - int(k==kre+1)
        p0[k-1]=p0[k]*A[k] + dV*B[k]*tau*j0[k]

    #############################################
    # first need the steady state
    #############################################


    r0=1./(dV*np.sum(p0))  # steady-state firing rate (in kHz)
    P0=r0*p0           # correctly normalised density and current
    J0=r0*j0



    ############################################
    # now the response to current (E) modulation
    ############################################

    #w=2*np.pi*fkHz

    jh0=np.zeros(n,dtype=complex)
    ph0=np.zeros(n,dtype=complex)

    jhr=np.zeros(n,dtype=complex)
    jhr[n-1]=1
    phr=np.zeros(n,dtype=complex)

    for k in np.arange(n-1, 0,-1):    # integrate backwards from threshold

        jh0[k-1]=jh0[k] + dV*1j*w*ph0[k] - np.exp(-1j*w*tref)*int(k==kre+1)
        ph0[k-1]=ph0[k]*A[k] + dV*B[k]*tau*jh0[k]

        jhr[k-1]=jhr[k] + dV*1j*w*phr[k]
        phr[k-1]=phr[k]*A[k] + dV*B[k]*tau*jhr[k]


    fh=-jh0[0]/jhr[0]       # because Jh(Vl)=0 => jhe(1) + r1*jh1(1)=0
    
    rho = 0. * 1j
    spikecov = 0. * 1j
    
    if(w!=0):
        rho = fh/(1.-fh) 
    else:
        rho=rho+np.pi*r0/dw0;          # this requires a dw to be correctly defined

    spikecov=r0*(1.+2.*np.real(rho))

    return V,P0,J0,r0, fh , rho, spikecov



