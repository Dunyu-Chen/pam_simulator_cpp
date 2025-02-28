#include "pam_simulator_cpp/lib_simulator.h"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include <functional>
#include <chrono>
#include <memory>

class Rigid_Tank_Node : public rclcpp::Node
{
public:
    Rigid_Tank_Simulator simulator;
    Rigid_Tank_Node(double volume_0,Pneumatic_System_Constants constants,double value_b,double value_c, double sample_time_s) :
        rclcpp::Node("Rigid_Tank_Node"),
        simulator(volume_0,constants,value_b,value_c,sample_time_s),
        timer_(
            this->create_wall_timer(
                std::chrono::milliseconds(static_cast<int>(sample_time_s * 1e3)),
                std::bind(&Rigid_Tank_Node::timer_callback,this)
            )
        ),
        publisher_(this->create_publisher<std_msgs::msg::Float64>("pressure", 10)),
        subscription_(
            this->create_subscription<std_msgs::msg::Float64>(
                "input",10,
                std::bind(
                    &Rigid_Tank_Node::subscription_callback,this,std::placeholders::_1
                )
            )
        )
    {}
private:
    void subscription_callback(const std_msgs::msg::Float64& sub_msg)
    {
        if (sub_msg.data>0)
        {
            simulator.input_vector << 1,0;
        }
        else if (sub_msg.data==0)
        {
            simulator.input_vector << 0,0;
        }
        else if (sub_msg.data<0)
        {
            simulator.input_vector << 0,1;
        }
    }
    void timer_callback()
    {
        std_msgs::msg::Float64 publish_msg;
        double pressure = simulator.step(simulator.input_vector)[0];
        publish_msg.data = pressure;
        publisher_->publish(publish_msg);
    }
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr publisher_;
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr subscription_;
};
int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    double sample_time = 0.01; // 10ms = 0.01s
    Pneumatic_System_Constants constants;
    constants.pressure_supply = 5.0 * 1e5;// 500 KPa = 5 bar
    constants.pressure_air = 0.0; // 0 Pa
    double volume_0 = 1.0 * 1e-3;// 1 L = 1e-3 m^3
    double valve_b =0.38; // ratio
    double valve_c =0.4*1e-8; // 0.4 L/sbar = 0.4 * 1e-8 m^3/sPa
    Eigen::VectorXd init_state(1);
    init_state<<0.0;
    Eigen::VectorXd init_input(2);
    init_input<< 0,0;

    auto node_ = std::make_shared<Rigid_Tank_Node>(
        volume_0,constants,valve_b,valve_c,sample_time
        );
    node_->simulator.reset(init_input,init_state);
    RCLCPP_INFO(node_->get_logger(),"Starting rigid tank node");
    rclcpp::spin(node_);
    rclcpp::shutdown();
    return 0;
}

