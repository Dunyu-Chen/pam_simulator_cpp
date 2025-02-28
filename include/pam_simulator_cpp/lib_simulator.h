//
// Created by dunyu on 2/6/25.
//

#ifndef LIB_SIMULATOR_H
#define LIB_SIMULATOR_H

#include "Eigen/Dense"
struct Pneumatic_System_Constants
{
    double air_density_ref = 1.185; // kg/m^3
    double gas_constant = 287.0; //J/kg/K
    double air_temp_ref = 293.15; // K
    double air_temp_supply = 293.15; // K
    double pressure_supply = 1.0 * 1e5; // [Pa]
    double pressure_air=0.0; // Pa
};
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
    Solenoid_Valve_2_2 inlet_valve,outlet_valve;
    double alpha_in, alpha_out, alpha_thermal;
    double pressure;
    double pressure_dot;
    double volume;
    double volume_dot;
    Eigen::Vector<double,3> params;
    Valve_2_2_Controlled_Volume(Pneumatic_System_Constants& Constants,double B_value, double C_value) :
        constants(Constants),
        inlet_valve(constants,B_value,C_value), outlet_valve(constants,B_value,C_value),
        alpha_in(1.4),alpha_out(1.0),alpha_thermal(1.2),
        pressure(constants.pressure_air),pressure_dot(0.0),
        volume(1.0),volume_dot(0.0),
        params(Eigen::Vector3d::Zero())
    {}
    Eigen::Vector<double,3> calculate_params(double Pressure,double Volume,double Volume_dot)
    {
        pressure = Pressure;
        volume = Volume;
        volume_dot = Volume_dot;
        double mid_term = (constants.gas_constant * constants.air_temp_supply) / volume;
        params[0] = mid_term * alpha_in * inlet_valve.calculate_flow(constants.pressure_supply,pressure);
        params[1] = - mid_term * alpha_out * outlet_valve.calculate_flow(pressure,constants.pressure_air);
        params[2] = alpha_thermal * (volume_dot)/(volume);
        return params;
    }
    double calculate_pressure_dot(double Pressure,double Volume,double Volume_dot,Eigen::VectorXd& Valve_input)
    {
        assert(Valve_input.rows() == 2 && Valve_input.cols() == 1);
        calculate_params(Pressure,Volume,Volume_dot);
        Eigen::Vector3d state_extended;
        state_extended<< Valve_input[0],Valve_input[1],pressure;
        pressure_dot = (params.transpose() * state_extended)[0];
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
};
/*
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


class Control_Affine : public Forward_Euler_Simulator
{
public:
    Eigen::MatrixXd A_matrix;
    Eigen::MatrixXd B_matrix;
    Eigen::VectorXd D_matrix;
    Control_Affine(size_t State_dim,size_t Input_dim,double Sample_time_s) :
        Forward_Euler_Simulator(State_dim,Input_dim,Sample_time_s),
        A_matrix(Eigen::MatrixXd::Zero(state_dim,input_dim)),
        B_matrix(Eigen::MatrixXd::Zero(state_dim,input_dim)),
        D_matrix(Eigen::VectorXd::Zero(state_dim))
    {}

    void update_params(Eigen::MatrixXd& A,Eigen::MatrixXd& B, Eigen::VectorXd& D)
    {
        assert(A.rows() == state_dim && A.cols() == state_dim);
        assert(B.rows() == state_dim && B.cols() == input_dim);
        assert(D.rows() == state_dim);
        A_matrix = A;
        B_matrix = B;
        D_matrix = D;
    }

    Eigen::VectorXd calculate_state_dot(Eigen::VectorXd Input) override
    {
        assert(Input.rows() == input_dim);
        dot_state_vector = A_matrix * state_vector + B_matrix * input_vector + D_matrix;
        return dot_state_vector;
    }
};

class Solenoid_Valve_Controlled_Volume
{
public:
    Solenoid_Valve_2_2 inlet_valve, outlet_valve;
    double gas_constant;
    double temp_supply;
    double pressure_supply; double pressure_air;
    double volume;
    double volume_dot;
    double pressure_chamber;
    double pressure_chamber_dot;
    double alpha_in, alpha_out, alpha_thermal;
    std::vector<double> params;

    Solenoid_Valve_Controlled_Volume(double Pressure_supply,double Valve_B, double Valve_C) :
        inlet_valve(Valve_B, Valve_C), outlet_valve(Valve_B, Valve_C),
        gas_constant(287.0),
        temp_supply(293.15),
        pressure_air(0.0),pressure_supply(Pressure_supply),
        volume(1.0), volume_dot(0.0),
        pressure_chamber(0.0), pressure_chamber_dot(0.0),
        alpha_in(1.4),alpha_out(1.0),alpha_thermal(1.2),
        params({0.0,0.0,0.0})
        {}

    std::vector<double> update_params ( double Pressure,double Volume, double Volume_dot)
    {
        pressure_chamber = Pressure;
        volume = Volume;
        volume_dot = Volume_dot;
        //
        params[0] = ( (gas_constant * temp_supply) / volume ) *
            alpha_in * inlet_valve.calculate_flow(pressure_supply,pressure_chamber);
        params[1] = ( (gas_constant * temp_supply) / volume ) *
            alpha_out * outlet_valve.calculate_flow(pressure_chamber,pressure_air);
        params[2] = - ( alpha_thermal * pressure_chamber * volume_dot) / volume ;
        return params;
    }
};

class Valve_Controlled_Rigid_Tank_Simulator : public Forward_Euler_Simulator
{
public:

    Solenoid_Valve_Controlled_Volume controlled_volume;

    Valve_Controlled_Rigid_Tank_Simulator(
        double Pressure_supply, double Pressure_air,
        double Volume,
        double Valve_B, double Valve_C,
        double Sample_time_s)
        :
        Forward_Euler_Simulator(3,2,Sample_time_s),
        controlled_volume(Valve_B,Valve_C)
    {
        controlled_volume.pressure_supply = Pressure_supply;
        controlled_volume.pressure_air = Pressure_air;
        controlled_volume.volume = Volume;
    }

    Eigen::VectorXd calculate_state_dot(Eigen::VectorXd Input)
    {
        assert(Input.rows() == input_dim);
        //
        controlled_volume.update_params(state_vector[0],controlled_volume.volume,controlled_volume.volume_dot);
        dot_state_vector[0] = Input[0] * controlled_volume.params[0] + Input[1] * controlled_volume.params[1] + controlled_volume.params[2];
        return dot_state_vector;
    }
};

class Single_Acting_Cylinder_Simulator : public Forward_Euler_Simulator
{
public:
    Solenoid_Valve_Controlled_Volume  controlled_volume;
    double Area;

    Single_Acting_Cylinder_Simulator(double Pressure_air,double Pressure_supply,double B, double C, double Area,double Sample_time) :
        Forward_Euler_Simulator(3,2,Sample_time),
        controlled_volume(B,C),
        Area(Area)
        {
            controlled_volume.pressure_supply = Pressure_supply;
            controlled_volume.pressure_air = Pressure_air;
        }
};*/

/*class Rigid_Tank_Simulator : public Forward_Euler_Simulator
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
};*/
#endif //LIB_SIMULATOR_H
