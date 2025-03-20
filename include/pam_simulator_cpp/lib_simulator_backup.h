#ifndef LIB_SIM_H
#define LIB_SIM_H

#include "Eigen/Dense"
#include "torch/torch.h"

inline torch::Tensor test()
{
    return torch::randn({1,1});
}

struct Pneumatic_System_Constants
{
    double air_density_ref = 1.185; // kg/m^3
    double gas_constant = 287.0; //J/kg/K
    double air_temp_ref = 293.15; // K
    double air_temp_supply = 293.15; // K
    double pressure_supply = 1.0 * 1e5; // [Pa]
    double pressure_air=0.0; // Pa
};

inline struct Valve_Params
{
    double b_value = 0.38;
    double c_value = 0.4*1e-8;
}MHJ10_Params;

class Solenoid_Valve_2_2
{
public:
    Pneumatic_System_Constants constants;
    double b_value; // critical_ratio
    double c_value; // sonic_conductance [m^3/sPa]
    double mass_flow; // [kg/s]
    Solenoid_Valve_2_2(Pneumatic_System_Constants& Constants,double B_value, double C_value):
        constants(Constants),
        b_value(B_value),
        c_value(C_value),
        mass_flow(0.0)
    {};
    double calculate_flow(double Pressure_in,double Pressure_out)
    {
        double mid_term = Pressure_in * c_value * constants.air_density_ref
            * sqrt(constants.air_temp_ref/constants.air_temp_supply) ;

        double pressure_ratio = Pressure_out/Pressure_in;
        if (pressure_ratio > 1)
        {
            pressure_ratio = 1.0;
        }
        if (pressure_ratio < 0)
        {
            pressure_ratio = 0.0;
        }

        if (pressure_ratio > b_value)
        {
            // subsonic regime

            mass_flow = mid_term
                * sqrt(1-pow((pressure_ratio-b_value)/(1-b_value),2));
        }
        else if (pressure_ratio <= b_value)
        {
            // choked regime
            mass_flow = mid_term;
        }
        return mass_flow;
    }
};
class Valve_2_2_Controlled_Volume
{
public:
    Pneumatic_System_Constants constants;
    Valve_Params valve_params;
    Solenoid_Valve_2_2 inlet_valve,outlet_valve;
    double alpha_in, alpha_out, alpha_thermal;
    Eigen::Vector<double,3> params;
    Valve_2_2_Controlled_Volume(Pneumatic_System_Constants& Constants,Valve_Params Valve_params) :
        constants(Constants),
        valve_params(Valve_params),
        inlet_valve(constants,valve_params.b_value,valve_params.c_value),
        outlet_valve(constants,valve_params.b_value,valve_params.c_value),
        alpha_in(1.4),alpha_out(1.0),alpha_thermal(1.2),
        params(Eigen::Vector3d::Zero())
    {}
    Eigen::Vector<double,3> calculate_params(double Pressure,double Volume,double Volume_dot)
    {
        //pressure = Pressure;
        //volume = Volume;
        //volume_dot = Volume_dot;
        double mid_term = (constants.gas_constant * constants.air_temp_supply) / Volume;
        params[0] = mid_term * alpha_in * inlet_valve.calculate_flow(constants.pressure_supply,Pressure);
        params[1] = - mid_term * alpha_out * outlet_valve.calculate_flow(Pressure,constants.pressure_air);
        params[2] = - alpha_thermal * (Volume_dot)/(Volume);
        return params;
    }
    double calculate_pressure_dot(double Pressure,double Volume,double Volume_dot,Eigen::VectorXd& Valve_input)
    {
        assert(Valve_input.rows() == 2 && Valve_input.cols() == 1);
        calculate_params(Pressure,Volume,Volume_dot);
        Eigen::Vector3d state_extended;
        state_extended<< Valve_input[0],Valve_input[1],Pressure+1.01e5;
        double pressure_dot = (params.transpose() * state_extended)[0];
        return pressure_dot;
    }
};

class Simulator
{
public:
    uint input_dim;
    uint state_dim;
    double sample_time_s;
    Eigen::VectorXd input_vector;
    Eigen::VectorXd state_vector;
    Simulator(uint Input_dim,uint State_dim,double Sample_time_s) :
        input_dim(Input_dim),
        state_dim(State_dim),
        sample_time_s(Sample_time_s),
        input_vector(Eigen::VectorXd::Zero(input_dim)),
        state_vector(Eigen::VectorXd::Zero(state_dim))
    {}

    virtual ~Simulator()= default;

    virtual void reset(Eigen::VectorXd& Init_input,Eigen::VectorXd& Init_state)
    {
        assert(Init_state.rows() == state_dim &&  Init_state.cols() == 1);
        assert(Init_input.rows() == input_dim &&  Init_input.cols() == 1);
        input_vector = Init_input;
        state_vector = Init_state;
    }
    virtual Eigen::VectorXd& step(Eigen::VectorXd& Input)
    {
        assert(Input.rows() == state_dim &&  Input.cols() == 1);
        input_vector = Input;
        return state_vector;
    }
};
class Forward_Euler_Simulator : public Simulator
{
public:
    Eigen::VectorXd state_dot;
    Forward_Euler_Simulator(uint Input_dim,uint State_dim,double Sample_time_s) :
        Simulator(Input_dim,State_dim,Sample_time_s),
        state_dot(Eigen::VectorXd::Zero(state_dim))
    {}

    virtual void calculate_state_dot(Eigen::VectorXd& Input)
    {
        input_vector = Input;
    };

    Eigen::VectorXd& step(Eigen::VectorXd& Input) override
    {
        assert(Input.rows() == input_dim &&  Input.cols() == 1);
        input_vector = Input;
        calculate_state_dot(input_vector);
        assert(state_dot.rows() == state_dim && state_dot.cols() == 1);
        state_vector = sample_time_s * state_dot + state_vector;
        return state_vector;
    }
};

/*
class Rigid_Tank_Simulator : public Forward_Euler_Simulator
{
public:
    double volume;
    Valve_2_2_Controlled_Volume chamber;
    Rigid_Tank_Simulator(double Volume, Pneumatic_System_Constants& Constants,double B_value,double C_value,double Sample_time_s) :
    Forward_Euler_Simulator(2,1,Sample_time_s),
    volume(Volume),
    chamber(Constants,B_value,C_value)
    {
        Eigen::VectorXd init_state(1);
        init_state << chamber.constants.pressure_air;
        Eigen::VectorXd init_input(2);
        init_input << 0,0;
        Rigid_Tank_Simulator::reset(init_input,init_state);
    }

    void reset(Eigen::VectorXd& Init_input,Eigen::VectorXd& Init_state) override
    {
        Simulator::reset(Init_input,Init_state);
        chamber.volume = volume;
        chamber.volume_dot = 0.0;
        chamber.pressure = state_vector[0];
    }

    void calculate_state_dot(Eigen::VectorXd& Input) override
    {
        state_dot[0] = chamber.calculate_pressure_dot(state_vector[0],chamber.volume,chamber.volume_dot,Input);
    }
};*/

class Linear_3Element_Actuator_Sim : public Forward_Euler_Simulator
{
public:
    Eigen::Vector3d three_elements;
    Eigen::Vector2d volume_vector;
    double inertia_mass;
    double external_load;
    Eigen::Vector3d observe_vector;
    Valve_2_2_Controlled_Volume chamber;
    Linear_3Element_Actuator_Sim(Pneumatic_System_Constants Constants,Valve_Params valve_params,double Sample_time_s) :
        Forward_Euler_Simulator(2,3,Sample_time_s),
        three_elements(Eigen::Vector3d::Zero()),
        volume_vector(Eigen::Vector2d::Zero()),
        inertia_mass(1.0),
        external_load(0.0),
        observe_vector(Eigen::Vector3d::Zero()),
        chamber(Constants,valve_params)
    {
        state_vector<<0.0,0.0,chamber.constants.pressure_air;
        observe_vector<<
            state_vector[0],
            state_vector[2],
            inertia_mass * state_dot(1) - external_load;
    }
    virtual Eigen::Vector3d calculate_3_elements (Eigen::VectorXd& State_vector)
    {
        assert(State_vector.rows() == state_dim);
        Eigen::Vector3d elements=Eigen::Vector3d::Zero();
        return elements;
    }
    virtual Eigen::Vector2d calculate_volume_dynamics (Eigen::VectorXd& State_vector)
    {
        assert(State_vector.rows() == state_dim);
        Eigen::Vector2d volume_vector=Eigen::Vector2d::Zero();
        return volume_vector;
    }
    Eigen::Vector3d observe()
    {
        observe_vector<<
            state_vector[0],
            state_vector[2],
            state_dot[1] * inertia_mass - external_load;
        return observe_vector;
    }
    void calculate_state_dot(Eigen::VectorXd& Input) override
    {
        volume_vector=calculate_volume_dynamics(this->state_vector);
        three_elements = calculate_3_elements(this->state_vector);
        state_dot[0] = state_vector[1];
        state_dot[1] = (1.0/inertia_mass) * ((three_elements.transpose() * state_vector)[0] + external_load);
        state_dot[2] = chamber.calculate_pressure_dot(state_vector[2],volume_vector[0],volume_vector[1],Input);
    }
};

class Rigid_Tank_Sim : public Linear_3Element_Actuator_Sim
{
public:
    double volume_0;
    Rigid_Tank_Sim (Pneumatic_System_Constants Constants,Valve_Params valve_params,double Sample_time_s) :
    Linear_3Element_Actuator_Sim(Constants,valve_params,Sample_time_s),
    volume_0(1.0)
    {}
    Eigen::Vector3d calculate_3_elements (Eigen::VectorXd& State_vector) override
    {
        assert(State_vector.rows() == state_dim);
        Eigen::Vector3d elements=Eigen::Vector3d::Zero();
        return elements;
    }
    Eigen::Vector2d calculate_volume_dynamics (Eigen::VectorXd& State_vector) override
    {
        assert(State_vector.rows() == state_dim);
        Eigen::Vector2d volume_vector=Eigen::Vector2d::Zero();
        volume_vector[0] = volume_0;
        return volume_vector;
    }
};

class Single_Effect_Cylinder_Sim : public Linear_3Element_Actuator_Sim
{
public:
    double volume_0;
    double cylinder_area;
    Single_Effect_Cylinder_Sim (double Volume_0,double Area,Pneumatic_System_Constants Constants, Valve_Params valve_params, double Sample_time_s) :
        Linear_3Element_Actuator_Sim(Constants,valve_params,Sample_time_s),
        volume_0(Volume_0),
        cylinder_area(Area)
    {
    }
    Eigen::Vector3d calculate_3_elements (Eigen::VectorXd& State_vector) override
    {
        assert(State_vector.rows() == state_dim);
        Eigen::Vector3d elements=Eigen::Vector3d::Zero();
        elements[0] = 0.0; // k spring
        elements[1] = 0.0; // b damping
        elements[2] = cylinder_area;
        return elements;
    }
    Eigen::Vector2d calculate_volume_dynamics (Eigen::VectorXd& State_vector) override
    {
        assert(State_vector.rows() == state_dim);
        Eigen::Vector2d volume_vector=Eigen::Vector2d::Zero();
        volume_vector[0] = volume_0 + cylinder_area * State_vector[0];
        if (volume_vector[0] <= 0.0){volume_vector[0] = 0.0;}
        volume_vector[1] = cylinder_area * State_vector[1];
        return volume_vector;
    }
};

#endif //LIB_SIM_H
