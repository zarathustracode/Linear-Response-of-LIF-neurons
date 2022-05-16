#include <cmath>
#include <algorithm>    // std::find_if

#include "LIF0.hpp"

//#include <iostream>
//#include <Eigen/Dense>
//using namespace Eigen;

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

//using namespace boost::python;

//using namespace std::complex_literals;

namespace py = boost::python;
namespace np = boost::python::numpy;

template<typename T>
std::vector<T> arange(T start, T stop, T step = 1) {
    std::vector<T> values;
    for (T value = start; value < stop; value += step)
        values.push_back(value);
    return values;
}


//constexpr unsigned long power(unsigned long base, unsigned long exponent) {
//    return exponent == 0 ? 1 : base * pow(base, exponent - 1);
//}

//const unsigned long imax = power(2, 16);

const double pi = 3.14159265358979323846;

const std::complex<double> zero(0., 0.);
const std::complex<double> rone(1., 0.);
const std::complex<double> jone(0., 1.);


char const* help()
{
   const char* s1 = R"(#                                     LIF0
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
    #             scalar r0 [kHz] steady state firing rate;)";

    return s1;
}

std::vector<double> lif(double mu,  double sig,  double Vresting, double Vth, double Vre, double tau,  double tau_ref)
{
    double E0 = Vresting + mu;
    
    //set up the lattice for the discretized integration
    double dV=0.01;
    double Vl=Vresting-20.;           // the lower bound for the voltage range

    auto V = arange<double>(Vl,Vth+dV,dV);
    auto n = V.size();

    double tol = 1.0e-10; // Edit this to change tolerance

    std::vector<double>::iterator it = std::find_if (V.begin(), V.end(), [Vre,tol] (const double& v1) { return  abs(v1-Vre)< tol; });
    // Get index of element from iterator
    int kre = std::distance(V.begin(), it);

    //kre=np.where( np.abs(V - Vre ) < tol )[0][0]   # NB Vre must fall on a lattice point!

    // Will use the modified Euler method (Phys. Rev. E 2007, Eqs A1-A6)
    std::vector<double> G = V;

    for(auto& element : G){
        element = 2.*(element - E0)/pow(sig,2);
    }

    std::vector<double> A = G;

    for(auto& element : A){
        element = exp(dV*element);
    }
    
    std::vector<double> B = A;

    for(std::size_t i = 0; i < n; ++i){
        if(G[i]==0) B[i] = 2./ pow(sig,2);
        else B[i] = (A[i]-1.)/(pow(sig,2)*dV*G[i]/2.);
    }


    // set up the vectors for the scaled current and probability density
    std::vector<double> j0(n,0);
    auto p0 = j0;

    j0[n-1]=1.;        // initial conditions at V=Vth. NB p0(n)=0 already.

    for(std::size_t k=n-1; k>0;--k){
        j0[k-1]=j0[k] - (int)(k==(kre+1));
        p0[k-1]=p0[k]*A[k] + dV*B[k]*tau*j0[k]; // ther is correction 2 for Brunel noise
    }

    double p0sum = 0.;

    for(auto& element : p0) {
        p0sum += element;
    }

    double r0=1./(tau_ref+dV*p0sum); // steady-state firing rate (in kHz)

    auto P0 = p0;
    auto J0 = j0;

    for(std::size_t i = 0; i < n; ++i){ // correctly normalised density and current
        P0[i] = r0*p0[i];
        J0[i] = r0*J0[i];
    }

    //    return V,P0,J0,r0
    return P0;

};

np::ndarray lifpy(double mu,  double sig,  double Vresting, double Vth, double Vre, double tau,  double tau_ref) {
    std::vector<double> v = lif(mu,  sig,  Vresting, Vth, Vre, tau,  tau_ref);
    Py_intptr_t shape[1] = { v.size() };
    np::ndarray result = np::zeros(1, shape, np::dtype::get_builtin<double>());
    std::copy(v.begin(), v.end(), reinterpret_cast<double*>(result.get_data()));
    return result;
}

std::tuple<double, std::vector<double>, std::vector<double>> lif1(double mu,  double sig,  double Vresting, double Vth, double Vre, double tau,  double tau_ref)
{
    double E0 = Vresting + mu;
    
    //set up the lattice for the discretized integration
    double dV=0.01;
    double Vl=Vresting-20.;           // the lower bound for the voltage range

    auto V = arange<double>(Vl,Vth+dV,dV);
    auto n = V.size();

    double tol = 1.0e-10; // Edit this to change tolerance

    std::vector<double>::iterator it = std::find_if (V.begin(), V.end(), [Vre,tol] (const double& v1) { return  abs(v1-Vre)< tol; });
    // Get index of element from iterator
    int kre = std::distance(V.begin(), it);

    //kre=np.where( np.abs(V - Vre ) < tol )[0][0]   # NB Vre must fall on a lattice point!

    // Will use the modified Euler method (Phys. Rev. E 2007, Eqs A1-A6)
    std::vector<double> G = V;

    for(auto& element : G){
        element = 2.*(element - E0)/pow(sig,2);
    }

    std::vector<double> A = G;

    for(auto& element : A){
        element = exp(dV*element);
    }
    
    std::vector<double> B = A;

    for(std::size_t i = 0; i < n; ++i){
        if(G[i]==0) B[i] = 2./ pow(sig,2);
        else B[i] = (A[i]-1.)/(pow(sig,2)*dV*G[i]/2.);
    }


    // set up the vectors for the scaled current and probability density
    std::vector<double> j0(n,0);
    auto p0 = j0;

    j0[n-1]=1.;        // initial conditions at V=Vth. NB p0(n)=0 already.

    for(std::size_t k=n-1; k>0;--k){
        j0[k-1]=j0[k] - (int)(k==(kre+1));
        p0[k-1]=p0[k]*A[k] + dV*B[k]*tau*j0[k]; // There is factor of two as correction for Brunel noise \sigma instead of \sqrt 2 \sigma
    }

    double p0sum = 0.;

    for(auto& element : p0) {
        p0sum += element;
    }

    double r0=1./(tau_ref+dV*p0sum); // steady-state firing rate (in kHz)

    auto P0 = p0;
    auto J0 = j0;

    for(std::size_t i = 0; i < n; ++i){ // correctly normalised density and current
        P0[i] = r0*p0[i];
        J0[i] = r0*J0[i];
    }

    //    return V,P0,J0,r0
    return {r0,V,P0};

};

boost::python::tuple lif1py(double mu,  double sig,  double Vresting, double Vth, double Vre, double tau,  double tau_ref) {

    auto [rate, v, p] = lif1(mu,  sig,  Vresting, Vth, Vre, tau,  tau_ref);

    Py_intptr_t shapev[1] = { v.size() };
    np::ndarray resultv = np::zeros(1, shapev, np::dtype::get_builtin<double>());
    std::copy(v.begin(), v.end(), reinterpret_cast<double*>(resultv.get_data()));

    Py_intptr_t shapep[1] = { p.size() };
    np::ndarray resultp = np::zeros(1, shapep, np::dtype::get_builtin<double>());
    std::copy(p.begin(), p.end(), reinterpret_cast<double*>(resultp.get_data()));

    return boost::python::make_tuple(rate, resultv, resultp);
}


std::tuple<double, std::vector<double>, std::vector<double>, std::vector <double>, std::complex <double> >
 lifResponse(double mu,  double sig,  double Vresting, double Vth, double Vre, double tau,  double tau_ref, double E1, double w, double dw0)
{
    double E0 = Vresting + mu;
    
    //set up the lattice for the discretized integration
    double dV=0.01;
    double Vl=Vresting-20.;           // the lower bound for the voltage range

    auto V = arange<double>(Vl,Vth+dV,dV);
    auto n = V.size();

    double tol = 1.0e-10; // Edit this to change tolerance

    std::vector<double>::iterator it = std::find_if (V.begin(), V.end(), [Vre,tol] (const double& v1) { return  abs(v1-Vre)< tol; });
    // Get index of element from iterator
    int kre = std::distance(V.begin(), it);

    //kre=np.where( np.abs(V - Vre ) < tol )[0][0]   # NB Vre must fall on a lattice point!

    // Will use the modified Euler method (Phys. Rev. E 2007, Eqs A1-A6)
    std::vector<double> G = V;

    for(auto& element : G){
        element = 2.*(element - E0)/pow(sig,2);
    }

    std::vector<double> A = G;

    for(auto& element : A){
        element = exp(dV*element);
    }
    
    std::vector<double> B = A;

    for(std::size_t i = 0; i < n; ++i){
        if(G[i]==0) B[i] = 2./ pow(sig,2);
        else B[i] = (A[i]-1.)/(pow(sig,2)*dV*G[i]/2.);
    }


    // set up the vectors for the scaled current and probability density
    std::vector<double> j0(n,0);
    auto p0 = j0;

    j0[n-1]=1.;        // initial conditions at V=Vth. NB p0(n)=0 already.

    for(std::size_t k=n-1; k>0;--k){
        j0[k-1]=j0[k] - (int)(k==(kre+1));
        p0[k-1]=p0[k]*A[k] + dV*B[k]*tau*j0[k]; // There is factor of two as correction for Brunel noise \sigma instead of \sqrt 2 \sigma
    }

    double p0sum = 0.;

    for(auto& element : p0) {
        p0sum += element;
    }

    double r0=1./(tau_ref+dV*p0sum); // steady-state firing rate (in kHz)

    auto P0 = p0;
    auto J0 = j0;

    for(std::size_t i = 0; i < n; ++i){ // correctly normalised density and current
        P0[i] = r0*p0[i];
        J0[i] = r0*J0[i];
    }

    //############################################
    //# now the response to current (E) modulation
    //############################################

    std::vector<std::complex<double>> jh1(n,zero);
    auto ph1 = jh1;
    auto jhE = jh1;
    auto phE = jh1;

    auto r1 = zero;
    jh1[n-1]=rone;
    
    if(w!=0){

        for(std::size_t k=n-1; k>0;--k){
        //double resetcondition = (k==(kre+1))?1.0:0.0;
        //auto resetcondition = (double) (k == (kre+1))
        //auto reset = std::complex<double>(resetcondition, 0.);
        jh1[k-1]=jh1[k] + dV*jone*w*ph1[k] - (double) (k == (kre+1));
        ph1[k-1]=ph1[k]*A[k] + dV*B[k]*tau*jh1[k]; 

        jhE[k-1]=jhE[k] + dV*jone*w*phE[k];
        phE[k-1]=phE[k]*A[k] + dV*B[k]*(tau*jhE[k] - E1*P0[k]);

        }
        r1=-jhE[0]/jh1[0];       // because Jh(Vl)=0 => jhe(1) + r1*jh1(1)=0
    };  

    //    return V,P0,J0,r0 V,P0,J0,r0, r1

    return {r0,V,P0, J0, r1};

};

boost::python::tuple 
lifResponsepy(double mu,  double sig,  double Vresting, double Vth, double Vre, double tau,  double tau_ref, double E1, double w, double dw0) {

    auto [rate, v, p, j, rmod] = lifResponse(mu,  sig,  Vresting, Vth, Vre, tau,  tau_ref, E1, w, dw0);

    Py_intptr_t shapev[1] = { v.size() };
    np::ndarray resultv = np::zeros(1, shapev, np::dtype::get_builtin<double>());
    std::copy(v.begin(), v.end(), reinterpret_cast<double*>(resultv.get_data()));

    Py_intptr_t shapep[1] = { p.size() };
    np::ndarray resultp = np::zeros(1, shapep, np::dtype::get_builtin<double>());
    std::copy(p.begin(), p.end(), reinterpret_cast<double*>(resultp.get_data()));

    Py_intptr_t shapej[1] = { j.size() };
    np::ndarray resultj = np::zeros(1, shapep, np::dtype::get_builtin<double>());
    std::copy(j.begin(), j.end(), reinterpret_cast<double*>(resultj.get_data()));

    return boost::python::make_tuple(rate, resultv, resultp, resultj, rmod);
}

std::tuple<double, std::vector<double>, std::vector<double>, std::vector <double>, std::complex <double> , std::complex<double> >
 lifStatistics(double mu,  double sig,  double Vresting, double Vth, double Vre, double tau,  double tau_ref, double w, double dw0)
{
    double E0 = Vresting + mu;
    
    //set up the lattice for the discretized integration
    double dV=0.01;
    double Vl=Vresting-20.;           // the lower bound for the voltage range

    auto V = arange<double>(Vl,Vth+dV,dV);
    auto n = V.size();

    double tol = 1.0e-10; // Edit this to change tolerance

    std::vector<double>::iterator it = std::find_if (V.begin(), V.end(), [Vre,tol] (const double& v1) { return  abs(v1-Vre)< tol; });
    // Get index of element from iterator
    int kre = std::distance(V.begin(), it);

    //kre=np.where( np.abs(V - Vre ) < tol )[0][0]   # NB Vre must fall on a lattice point!

    // Will use the modified Euler method (Phys. Rev. E 2007, Eqs A1-A6)
    std::vector<double> G = V;

    for(auto& element : G){
        element = 2.*(element - E0)/pow(sig,2);
    }

    std::vector<double> A = G;

    for(auto& element : A){
        element = exp(dV*element);
    }
    
    std::vector<double> B = A;

    for(std::size_t i = 0; i < n; ++i){
        if(G[i]==0) B[i] = 2./ pow(sig,2);
        else B[i] = (A[i]-1.)/(pow(sig,2)*dV*G[i]/2.);
    }


    // set up the vectors for the scaled current and probability density
    std::vector<double> j0(n,0);
    auto p0 = j0;

    j0[n-1]=1.;        // initial conditions at V=Vth. NB p0(n)=0 already.

    for(std::size_t k=n-1; k>0;--k){
        j0[k-1]=j0[k] - (int)(k==(kre+1));
        p0[k-1]=p0[k]*A[k] + dV*B[k]*tau*j0[k]; // There is factor of two as correction for Brunel noise \sigma instead of \sqrt 2 \sigma
    }

    double p0sum = 0.;

    for(auto& element : p0) {
        p0sum += element;
    }

    double r0=1./(tau_ref+dV*p0sum); // steady-state firing rate (in kHz)

    auto P0 = p0;
    auto J0 = j0;

    for(std::size_t i = 0; i < n; ++i){ // correctly normalised density and current
        P0[i] = r0*p0[i];
        J0[i] = r0*J0[i];
    }

   //############################################
    //# now the spike statistics
    //############################################

    std::vector<std::complex<double>> jh0(n,zero);
    auto ph0 = jh0;
    auto jhr = jh0;
    auto phr = jh0;

    auto fh = zero;
    jhr[n-1]=rone;


    for(std::size_t k=n-1; k>0;--k){
        jh0[k-1]=jh0[k] + dV*jone*w*ph0[k] - std::exp(-jone*w*tau_ref)*(double) (k == (kre+1));
        ph0[k-1]=ph0[k]*A[k] + dV*B[k]*tau*jh0[k];

        jhr[k-1]=jhr[k] + dV*jone*w*phr[k];
        phr[k-1]=phr[k]*A[k] + dV*B[k]*tau*jhr[k];

    }

    fh=-jh0[0]/jhr[0];       // because Jh(Vl)=0 => jhe(1) + r1*jh1(1)=0
    
    auto rho = zero;
    auto cov = zero;
    
    if(w!=0) rho = fh/(1.-fh);
    else rho=rho+ pi*r0/dw0;          //this requires a dw to be correctly defined

    cov=r0*(1.+2.*std::real(rho));

    //    return V,P0,J0,r0 V,P0,J0,r0, r1

    return {r0,V,P0, J0, rho, cov};

};

boost::python::tuple 
lifStatisticspy(double mu,  double sig,  double Vresting, double Vth, double Vre, double tau,  double tau_ref, double w, double dw0) {

    auto [rate, v, p, j, rho, cov] = lifStatistics(mu,  sig,  Vresting, Vth, Vre, tau,  tau_ref, w, dw0);

    Py_intptr_t shapev[1] = { v.size() };
    np::ndarray resultv = np::zeros(1, shapev, np::dtype::get_builtin<double>());
    std::copy(v.begin(), v.end(), reinterpret_cast<double*>(resultv.get_data()));

    Py_intptr_t shapep[1] = { p.size() };
    np::ndarray resultp = np::zeros(1, shapep, np::dtype::get_builtin<double>());
    std::copy(p.begin(), p.end(), reinterpret_cast<double*>(resultp.get_data()));

    Py_intptr_t shapej[1] = { j.size() };
    np::ndarray resultj = np::zeros(1, shapep, np::dtype::get_builtin<double>());
    std::copy(j.begin(), j.end(), reinterpret_cast<double*>(resultj.get_data()));

    return boost::python::make_tuple(rate, resultv, resultp, resultj, rho, cov);
}

std::tuple < double, std::complex <double>, std::complex <double>, std::complex <double> , std::complex <double>> 
getSolution(double mu,  double sig,  double Vresting, double Vth, double Vre, double tau,  double tau_ref, double E1, double w, double dw0)
{
    double E0 = Vresting + mu;
    
    //set up the lattice for the discretized integration
    double dV=0.01;
    double Vl=Vresting-20.;           // the lower bound for the voltage range

    auto V = arange<double>(Vl,Vth+dV,dV);
    auto n = V.size();

    double tol = 1.0e-10; // Edit this to change tolerance

    std::vector<double>::iterator it = std::find_if (V.begin(), V.end(), [Vre,tol] (const double& v1) { return  abs(v1-Vre)< tol; });
    // Get index of element from iterator
    int kre = std::distance(V.begin(), it);

    //kre=np.where( np.abs(V - Vre ) < tol )[0][0]   # NB Vre must fall on a lattice point!

    // Will use the modified Euler method (Phys. Rev. E 2007, Eqs A1-A6)
    std::vector<double> G = V;

    for(auto& element : G){
        element = 2.*(element - E0)/pow(sig,2);
    }

    std::vector<double> A = G;

    for(auto& element : A){
        element = exp(dV*element);
    }
    
    std::vector<double> B = A;

    for(std::size_t i = 0; i < n; ++i){
        if(G[i]==0) B[i] = 2./ pow(sig,2);
        else B[i] = (A[i]-1.)/(pow(sig,2)*dV*G[i]/2.);
    }


    // set up the vectors for the scaled current and probability density
    std::vector<double> j0(n,0);
    auto p0 = j0;

    j0[n-1]=1.;        // initial conditions at V=Vth. NB p0(n)=0 already.

    for(std::size_t k=n-1; k>0;--k){
        j0[k-1]=j0[k] - (int)(k==(kre+1));
        p0[k-1]=p0[k]*A[k] + dV*B[k]*tau*j0[k]; // There is factor of two as correction for Brunel noise \sigma instead of \sqrt 2 \sigma
    }

    double p0sum = 0.;

    for(auto& element : p0) {
        p0sum += element;
    }

    double r0=1./(tau_ref+dV*p0sum); // steady-state firing rate (in kHz)

    auto P0 = p0;
    auto J0 = j0;

    for(std::size_t i = 0; i < n; ++i){ // correctly normalised density and current
        P0[i] = r0*p0[i];
        J0[i] = r0*J0[i];
    }

    //############################################
    //# now the response to current (E) modulation
    //############################################

    std::vector<std::complex<double>> jh1(n,zero);
    auto ph1 = jh1;
    auto jhE = jh1;
    auto phE = jh1;

    auto r1 = zero;
    jh1[n-1]=rone;
    
    if(w!=0){

        for(std::size_t k=n-1; k>0;--k){
        //double resetcondition = (k==(kre+1))?1.0:0.0;
        //auto resetcondition = (double) (k == (kre+1))
        //auto reset = std::complex<double>(resetcondition, 0.);
        jh1[k-1]=jh1[k] + dV*jone*w*ph1[k] - (double) (k == (kre+1));
        ph1[k-1]=ph1[k]*A[k] + dV*B[k]*tau*jh1[k]; 

        jhE[k-1]=jhE[k] + dV*jone*w*phE[k];
        phE[k-1]=phE[k]*A[k] + dV*B[k]*(tau*jhE[k] - E1*P0[k]);

        }
        r1=-jhE[0]/jh1[0];       // because Jh(Vl)=0 => jhe(1) + r1*jh1(1)=0
    };  

    //    return V,P0,J0,r0 V,P0,J0,r0, r1

   //############################################
    //# now the spike statistics
    //############################################

    std::vector<std::complex<double>> jh0(n,zero);
    auto ph0 = jh0;
    auto jhr = jh0;
    auto phr = jh0;

    auto fh = zero;
    jhr[n-1]=rone;


    for(std::size_t k=n-1; k>0;--k){
        jh0[k-1]=jh0[k] + dV*jone*w*ph0[k] - std::exp(-jone*w*tau_ref)*(double) (k == (kre+1));
        ph0[k-1]=ph0[k]*A[k] + dV*B[k]*tau*jh0[k];

        jhr[k-1]=jhr[k] + dV*jone*w*phr[k];
        phr[k-1]=phr[k]*A[k] + dV*B[k]*tau*jhr[k];

    }

    fh=-jh0[0]/jhr[0];       // because Jh(Vl)=0 => jhe(1) + r1*jh1(1)=0
    
    auto rho = zero;
    auto cov = zero;
    
    if(w!=0) rho = fh/(1.-fh);
    else rho=rho+ pi*r0/dw0;          //this requires a dw to be correctly defined

    cov=r0*(1.+2.*std::real(rho));


    return { r0, r1, fh, rho, cov };
}

boost::python::tuple 
getSolutionpy(double mu,  double sig,  double Vresting, double Vth, double Vre, double tau,  double tau_ref, double E1, double w, double dw0) {

    auto [r0, r1, fh, rho, cov] = getSolution(mu,  sig,  Vresting, Vth, Vre, tau,  tau_ref, E1, w, dw0);

    return boost::python::make_tuple(r0, r1, fh, rho, cov);
}



BOOST_PYTHON_MODULE(LIF0)
{
    np::initialize();
    py::def("help", help);
    py::def("lifpy", lifpy);
    py::def("lif1py", lif1py);
    py::def("lifResponsepy", lifResponsepy);
    py::def("lifStatisticspy", lifStatisticspy);
    py::def("getSolutionpy", getSolutionpy);
}