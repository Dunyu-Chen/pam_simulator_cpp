#include "pam_simulator_cpp/lib_deepLearning.h"

int main()
{
    Dynamic_RBF_Net test_net(2,4,1);
    torch::Tensor zero_input = torch::zeros({10,2}).unsqueeze(0); // {1,10,i}
    while(true)
    {
        std::cout<<test_net.forward(zero_input)<<std::endl;

    }

    return 0;
}