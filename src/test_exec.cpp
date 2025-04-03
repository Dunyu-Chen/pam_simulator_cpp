#include "pam_simulator_cpp/lib_deepLearning.h"
std::pair<torch::Tensor, torch::Tensor> generate_data(int Num_samples=100)
{
    auto x = torch::linspace(0, 1, Num_samples).unsqueeze(-1); // [100,1]
}
int main()
{


    return 0;
}