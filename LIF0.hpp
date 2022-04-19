#pragma once

#include <vector>
#include <tuple>
#include <complex>

using namespace std::complex_literals;

std::vector<double> lif(double, double, double, double, double, double, double);

std::tuple<double, std::vector<double>, std::vector<double>> lif1(double, double, double, double, double, double, double);

std::tuple<double, std::vector<double>, std::vector<double>, std::vector <double>, std::complex <double> >
lifResponse(double, double, double, double, double, double, double, double, double, double);

std::tuple<double, std::vector<double>, std::vector<double>, std::vector <double>, std::complex <double> , std::complex<double> >
lifStatistics(double, double, double, double, double, double, double, double, double);

std::tuple < double, std::complex <double>, std::complex <double>, std::complex <double> , std::complex <double>> 
getSolution(double mu,  double sig,  double Vresting, double Vth, double Vre, double tau,  double tau_ref, double E1, double w, double dw0);