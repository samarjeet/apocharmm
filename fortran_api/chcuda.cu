#include<iostream>
#include<memory>
#include<CharmmContext.h>
#include<ForceManager.h>

std::unique_ptr<CharmmContext> charmmContext;
std::unique_ptr<ForceManager> forceManager;

extern "C" void initialization_chcuda(){
  //charmmContext = std::make_unique<CharmmContext>(1000);
  //forceManager = std::make_unique<ForceManager>();
}
extern "C" void calculate_energy_chcuda(){
  std::cout << "Calculating energy from C++\n";  
} 

extern "C" void dynamics_chcuda() {
  std::cout << "Dynamics from C++ ";
  //std::cout << "numAtoms: " << charmmContext->getNumAtoms() << "\n";
}

extern "C" void minimization_chcuda() {
  std::cout << "minimizing from C++\n";  

}

