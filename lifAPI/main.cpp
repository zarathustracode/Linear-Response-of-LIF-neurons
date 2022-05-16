  
  
#include <iostream>
#include <chrono> 
#include "LIF0.hpp"
#include <vector>

int main()
{
    auto start = std::chrono::high_resolution_clock::now();
  


    auto distribution = lif(10.,5.,-60.,-40.,-60., 20.,2.); //(mu,  sig,  Vresting, Vth, Vre, tau,  tau_ref)

    auto [rate, v, p] = lif1(10.,5.,-60.,-40.,-60., 20.,2.);

    for (auto &element : distribution) std::cout << element << std::endl; 

    auto stop = std::chrono::high_resolution_clock::now();

    
    // Get duration. Substart timepoints to  
    // get durarion. To cast it to proper unit 
    // use duration cast method 
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); 

    std::cout<< "The rate is: " << rate * 1000. << " Hz" << std::endl;
  
    std::cout << "Time taken by function: " << duration.count()/1000. << " ms" << std::endl; 
  
    return 0;
  
  }