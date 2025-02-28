#include "pam_simulator_cpp/lib_simulator.h"
#include "rclcpp/rclcpp.hpp"

int main()
{
    double sample_time = 0.01; // 10ms = 0.01s
    Pneumatic_System_Constants constants;
    constants.pressure_supply = 5.0 * 1e5;// 500 KPa = 5 bar
    constants.pressure_air = 0.0; // 0 Pa
    double volume_0 = 1.0 * 1e-3;// 1 L = 1e-3 m^3
    double valve_b =0.38; // ratio
    double valve_c =0.4*1e-8; // 0.4 L/sbar = 0.4 * 1e-8 m^3/sPa
    Rigid_Tank_Simulator test_simulator(volume_0,constants,valve_b,valve_c,sample_time);

    Eigen::VectorXd hold_input(2);
    hold_input<< 0.0, 0.0;
    Eigen::VectorXd discharge_input(2);
    discharge_input<< 0.0, 1.0;

    Eigen::VectorXd init_state(1);
    init_state << 0.01 *1e5;
    test_simulator.reset(hold_input,init_state);

    Eigen::VectorXd charge_input(2);
    charge_input <<1.0, 0.0;
    while(1)
    {
        std::cout<< test_simulator.step(discharge_input)<< std::endl;
    }
    return 0;
}
