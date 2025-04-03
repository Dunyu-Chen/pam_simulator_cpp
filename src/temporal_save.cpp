#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include <functional>
#include <chrono>
#include <memory>
#include "pam_simulator_cpp/lib_simulator.h"



int main(int argc, char * argv[])
{
    double sample_time = 0.01;
    Pneumatic_System_Constants constants;
    constants.pressure_supply = 1.5e5;
    double volume_0 = 1.0 * 1e-3; // m^3 = 1e3L
    double area = 400 * 1e-6 ; // m^2 = 1e6 mm^2
    double mass = 100 * 1e-3 ; // kg = 1e3 g
    double max_length = 0.5;
    double min_length = -0.5;
    Cylinder_Sim test_class(area,max_length,min_length,volume_0,MHJ10params,constants,sample_time);
    Eigen::Vector<double,4> Init_states({0.0,0.0,volume_0,constants.pressure_air});
    test_class.reset(Init_states);
    Eigen::Vector<double,2> charging_inputs({1.0,0.0});
    Eigen::Vector<double,2> discharging_inputs({0.0,1.0});
    Eigen::Vector<double,2> holding_inputs({0.0,0.0});
    Eigen::Vector<double,4> Init_disturbance({0.0,0.0/mass,0.0,0.0});
    while(true)
    {
        for (int i = 0;i < 15;i++)
        {
            test_class.step(charging_inputs,Init_disturbance);
        }
        for (int i = 0;i < 15;i++)
        {
            test_class.step(holding_inputs,Init_disturbance);
        }
        for (int i = 0;i < 15;i++)
        {
            test_class.step(discharging_inputs,Init_disturbance);
        }
    }

    return 0;
}
