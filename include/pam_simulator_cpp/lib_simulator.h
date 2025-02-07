//
// Created by dunyu on 2/6/25.
//

#ifndef LIB_SIMULATOR_H
#define LIB_SIMULATOR_H

#include "Eigen/Dense"

class Solenoid_Valve_2_2
{
public:
    double mass_flow_rate;
    double reference_air_density;
    double critical_pressure_ratio;
    double sonic_conductance;
    double reference_air_temperature;
    double upstream_air_temperature;
    //
    Solenoid_Valve_2_2(double B_value,double C_value) :
    mass_flow_rate(0.0),
    reference_air_density(1.185),
    critical_pressure_ratio(B_value),
    sonic_conductance(C_value),
    reference_air_temperature(293.15),
    upstream_air_temperature(293.15)
    {}
    double calculate_flow(double Pressure_up, double Pressure_down)
    {
        double pressure_ratio = Pressure_down / Pressure_up;
        if (pressure_ratio <= critical_pressure_ratio)
        {
            mass_flow_rate = Pressure_up * sonic_conductance * reference_air_density *
                sqrt(reference_air_temperature/upstream_air_temperature);
        }
        else // ratio > critical value (b) -> subsonic
        {
            double param = (pressure_ratio - critical_pressure_ratio)/(1-critical_pressure_ratio);
            mass_flow_rate = Pressure_up * sonic_conductance * reference_air_density *
                sqrt(reference_air_temperature/upstream_air_temperature) *
                sqrt(1-(param * param));
        }
        return mass_flow_rate;
    }
};

class Forward_Euler_Simulator
{
public:

    Eigen::VectorXd state_vector;
    Eigen::VectorXd dot_state_vector;
    Eigen::VectorXd input_vector;
    double sample_time_s;
    size_t state_dim;
    size_t input_dim;

    Forward_Euler_Simulator(size_t State_dim, size_t Input_dim, double Sample_time_s):
        state_dim(State_dim),
        input_dim(Input_dim),
        sample_time_s(Sample_time_s),
        state_vector(Eigen::VectorXd::Zero(State_dim)),
        input_vector(Eigen::VectorXd::Zero(Input_dim))
    {}

    void reset(Eigen::VectorXd Init_state)
    {
        assert(Init_state.rows() == state_dim);
        state_vector = Init_state;
    }

    virtual Eigen::VectorXd calculate_state_dot(Eigen::VectorXd Input)
    {
        assert(Input.rows() == input_dim);
    }

    virtual Eigen::VectorXd step(Eigen::VectorXd Input)
    {
        calculate_state_dot(Input);
        state_vector = state_vector + dot_state_vector * sample_time_s;
        return state_vector;
    }

    virtual ~Forward_Euler_Simulator() = default;
};

class Rigid_Tank_Simulator : public Forward_Euler_Simulator
{
public:

    double ideal_air_constant ; // J/kg/K
    double supply_gas_temperature ; // K
    double pressure_supply;
    double chamber_volume; // m^3
    double mass_flow_in, mass_flow_out; // kg/s
    double alpha_in, alpha_out; // polytropic constant
    double mid_param;
    Solenoid_Valve_2_2 inlet_valve, outlet_valve;
    Eigen::Matrix<double,1,2> mid_matrix;

    Rigid_Tank_Simulator(double Valve_b,double Valve_c,double Volume,double Pressure_supply,double Sample_time_s) :
        Forward_Euler_Simulator(1, 2, Sample_time_s),
        // state = [P] input = [u_1,u_2]^T
        ideal_air_constant(287.0),
        supply_gas_temperature(293.15),
        pressure_supply(Pressure_supply),
        chamber_volume(Volume),
        alpha_in(1.4),alpha_out(1),
        mass_flow_in(0.0), mass_flow_out(0.0),
        inlet_valve(Valve_b, Valve_c), outlet_valve(Valve_b, Valve_c),
        mid_matrix(Eigen::Matrix<double, 1, 2>::Zero())
    {
        mid_param = (ideal_air_constant*supply_gas_temperature)/chamber_volume;
    }

    Eigen::VectorXd calculate_state_dot(Eigen::VectorXd Input) override
    {
        assert(Input.rows() == input_dim);
        input_vector = Input;
        mass_flow_in = inlet_valve.calculate_flow(pressure_supply,state_vector[0]);
        mass_flow_out = - outlet_valve.calculate_flow(state_vector[0],0.0);
        mid_matrix << alpha_in * mass_flow_in, alpha_out * mass_flow_out;
        this->dot_state_vector =
            mid_param * mid_matrix * input_vector;
        return this->dot_state_vector;
    }
};
#endif //LIB_SIMULATOR_H
