#ifndef LIB_SIM_H
#define LIB_SIM_H
#include "Eigen/Dense"
#include "torch/torch.h"

template <int state_dim, int input_dim>
class Simulator
{
public:
    explicit Simulator(const double Sample_time) :
        state_vector(Eigen::Vector<double,state_dim>::Zero()),
        state_dot(Eigen::Vector<double,state_dim>::Zero()),
        input_vector(Eigen::Vector<double,input_dim>::Zero()),
        sample_time_s(Sample_time)
    {
        reset();
    }

    virtual ~Simulator() = default;

    Eigen::Vector<double,state_dim> state_vector;
    Eigen::Vector<double,state_dim> state_dot;
    Eigen::Vector<double,input_dim> input_vector;
    double sample_time_s ;
    enum class SolverType {RK4, ForwardEuler};
    //
    void reset
    (
        const Eigen::Vector<double,state_dim>& Init_state = Eigen::Vector<double,state_dim>::Zero()
    )
    {
        this->input_vector = Eigen::Vector<double,input_dim>::Zero();
        this->state_vector = Init_state;
    }

    virtual Eigen::Vector<double,state_dim> cal_state_dot
    (
        const Eigen::Vector<double,input_dim>& Inputs,
        const Eigen::Vector<double,state_dim>& States,
        const Eigen::Vector<double,state_dim>& Disturbances
    )
    {
        Eigen::Vector<double,state_dim> state_dots = Disturbances;
        return state_dots;
    }

    Eigen::Vector<double,state_dim> step
    (
        const Eigen::Vector<double,input_dim>& Inputs,
        const Eigen::Vector<double,state_dim>& Disturbance,
        const SolverType solver = SolverType::RK4
    )
    {
        switch(solver)
        {
            case SolverType::RK4:
                return rk4_step(Inputs,Disturbance);
            case SolverType::ForwardEuler:
                return forward_euler_step(Inputs,Disturbance);
            default:
                return forward_euler_step(Inputs,Disturbance);
        }
    }

    Eigen::Vector<double,state_dim> rk4_step(
        const Eigen::Vector<double,input_dim>& Inputs,
        const Eigen::Vector<double,state_dim>& Disturbance
    )
    {
        this->input_vector = Inputs;

        std::vector< Eigen::Vector<double,state_dim> > K(4);
        K[0] = this->cal_state_dot(
            Inputs,
            state_vector,
            Disturbance);
        K[1] = this->cal_state_dot(
            Inputs,
            state_vector+(sample_time_s/2)*K[0],
            Disturbance);
        K[2] = this->cal_state_dot(
            Inputs,
            state_vector+(sample_time_s/2)*K[1],
            Disturbance);
        K[3] = this->cal_state_dot(
            Inputs,
            state_vector+(sample_time_s)*K[2],
            Disturbance);
        this->state_dot = (1.0/6.0) * (K[0]+2*K[1]+2*K[2]+K[3]);
        this->state_vector += this->state_dot*sample_time_s;
        return this->state_vector;
    }
    Eigen::Vector<double,state_dim> forward_euler_step(
        const Eigen::Vector<double,input_dim>& Inputs,
        const Eigen::Vector<double,state_dim>& Disturbance
    )
    {
        this->input_vector = Inputs;
        state_dot = cal_state_dot (Inputs,state_vector,Disturbance);
        state_vector += state_dot*sample_time_s;
        return state_vector;
    }
};

struct Pneumatic_System_Constants
{
    double air_density_ref = 1.185; // kg/m^3
    double gas_constant = 287.0; //J/kg/K
    double air_temp_ref = 293.15; // K
    double air_temp_supply = 293.15; // K
    double pressure_supply = 1.1 * 1e5; // [Pa]
    double pressure_air=1.1 * 1e5; // Pa
};

struct Binary_Valve_Params
{
    double b_value;
    double c_value;
};

inline Binary_Valve_Params MHJ10params
//C -> 0.4 l/sbar = 4 x 10^-9 m^3/spa
//B -> 0.38
{0.38,4e-9 };

class Binary_Valve
{
public:
    Binary_Valve_Params params;
    Pneumatic_System_Constants constants;
    Binary_Valve(const Binary_Valve_Params Params, const Pneumatic_System_Constants& Constants) :
        params(Params),
        constants(Constants)
    {}

    double calculate_massflow(const double Pressure_up, const double Pressure_down) const
    // absolute pressure
    {
        assert(Pressure_up > 0 && Pressure_down > 0);
        const double mid_term = Pressure_up * params.c_value * constants.air_density_ref
            * sqrt(constants.air_density_ref/constants.air_temp_supply);

        const double pressure_ratio = Pressure_down/Pressure_up;
        const double safe_pressure_ratio = std::clamp(pressure_ratio,0.0,1.0);

        double mass_flow = 0.0;
        if (safe_pressure_ratio > this->params.b_value)
        {
            mass_flow = mid_term *
                sqrt(
                    1.0 -
                    pow(
                        (safe_pressure_ratio - this->params.b_value)/(1-this->params.b_value),
                        2
                    )
                );
        }
        else if (safe_pressure_ratio <= this->params.b_value)
        {
            mass_flow = mid_term;
        }
        return mass_flow;
    }
};

class Valve_Controlled_Chamber
{
public:
    Pneumatic_System_Constants  constants;
    Binary_Valve_Params inlet_params,outlet_params;
    Binary_Valve inlet,outlet;
    double alpha_in, alpha_out, alpha_thermal;

    Valve_Controlled_Chamber(
        const Pneumatic_System_Constants& Constants,
        const Binary_Valve_Params Inlet_params,
        const Binary_Valve_Params Outlet_params
    ) :
        constants(Constants),
        inlet_params(Inlet_params), outlet_params(Outlet_params),
        inlet(inlet_params,constants),
        outlet(outlet_params,constants),
        alpha_in(1.4),alpha_out(1.0),alpha_thermal(1.2)
    {
    }

    std::vector<double> calculate_terms(const double Pressure, const double Volume, const double Volume_dot) const
    {
        const double mass_flow_in = inlet.calculate_massflow(constants.pressure_supply,Pressure);
        const double mass_flow_out = outlet.calculate_massflow(Pressure,constants.pressure_air);
        double term_in = (constants.gas_constant * constants.air_temp_supply / Volume) * alpha_in * mass_flow_in;
        double term_out= - (constants.gas_constant * constants.air_temp_supply / Volume) * alpha_out * mass_flow_out;
        double term_thermal = - alpha_thermal * Pressure * Volume_dot / Volume;
        return {term_in, term_out, term_thermal};
    }
    double calculate_pressure_dot (const uint Inlet_state, const uint Outlet_state, const double Pressure, const double Volume, const double Volume_dot) const
    // absolute pressure
    {
        const std::vector<double> terms = calculate_terms(Pressure,Volume,Volume_dot);
        double pressure_dot = terms[0]* Inlet_state + terms[1] * Outlet_state + terms[2];
        return pressure_dot;
    }
};

class Linear_Pneumatic_Actuator : public Simulator<4,2>
{
public:
    Pneumatic_System_Constants  constants;
    Valve_Controlled_Chamber chamber;
    double volume_0 ;
    double inertia_mass;
    // states = [displacement, velocity, volume, delta_pressure]
    Linear_Pneumatic_Actuator
    (
        const Binary_Valve_Params Inlet_params,
        const Binary_Valve_Params Outlet_params,
        const Pneumatic_System_Constants& Constants,
        const double Sample_time
    ) :
        Simulator(Sample_time),
        constants(Constants),
        chamber(constants,Inlet_params,Outlet_params) ,
        volume_0(1.0),inertia_mass(1.0)
    {}

    Eigen::Vector<double, 4> cal_state_dot
    (
        const Eigen::Vector<double, 2>& Inputs,
        const Eigen::Vector<double, 4>& States,
        const Eigen::Vector<double, 4>& Disturbances
    ) override
    {
        Eigen::Vector<double, 4> State_dots = Eigen::Vector<double, 4>::Zero();
        // velocity
        State_dots[0] = States[1];

        // acceleration
        auto elements = this->calculate_3_elements(States);
        State_dots[1] = (1.0/inertia_mass) * (elements[0]*States[0] + elements[1]*States[1] + elements[2]*States[3]);
        //volume
        auto volume_ = this->calculate_volume_dynamic(States[0],States[1]);
        // volume dot
        State_dots[2] = volume_[1];

        // pressure dot
        State_dots[3] = this->chamber.calculate_pressure_dot(
            static_cast<uint>(Inputs[0]),
            static_cast<uint>(Inputs[1]),
            States[3]+constants.pressure_air,
            volume_[0],
            volume_[1]
        );
        State_dots += Disturbances;
        return State_dots;
    }

    virtual std::vector<double> calculate_volume_dynamic(double displacement,double velocity) const
    {
        double volume = std::max(volume_0,1e-20);
        double volume_dot = 0.0;
        return {volume,volume_dot};
    }
    virtual std::vector<double> calculate_3_elements(const Eigen::Vector<double,4>& States) const
    {
        return {0.0,0.0,0.0};
    }

};

class Rigid_Tank_Sim : public  Linear_Pneumatic_Actuator
{
public:
    Rigid_Tank_Sim(
        const double Volume,
        const Binary_Valve_Params Inlet_params,
        const Binary_Valve_Params Outlet_params,
        const Pneumatic_System_Constants& Constants,
        const double Sample_time
    ) :
        Linear_Pneumatic_Actuator
        (
            Inlet_params,
            Outlet_params,
            Constants, Sample_time
        )
    {
        this->volume_0 = Volume;
    }

    std::vector<double> calculate_volume_dynamic(double displacement,double velocity) const override
    {
        // constant volume for rigid tank
        double volume = std::max(volume_0, 1e-20);
        double volume_dot = 0.0;
        return {volume, volume_dot};
    }
    std::vector<double> calculate_3_elements(const Eigen::Vector<double,4>& States) const override
    {
        std::vector<double> elements(3);
        elements[0] = 0.0;
        elements[1] = 0.0;
        elements[2] = 0.0;
        return elements;
    }
};

class Cylinder_Sim : public  Linear_Pneumatic_Actuator
{
public:
    double area;
    double max_length,min_length;
    Cylinder_Sim(
        double Area, double Max_length,double Min_length,double Volume_0,
        Binary_Valve_Params Inlet_valve, Binary_Valve_Params Outlet_valve,
        Pneumatic_System_Constants Constants,
        double Sample_time
    ) :
        Linear_Pneumatic_Actuator(
            Inlet_valve,
            Outlet_valve,
            Constants,
            Sample_time),
        area(Area),
        max_length(Max_length),min_length(Min_length)
    {
        this->volume_0 = Volume_0;
    }
    std::vector<double> calculate_volume_dynamic(double displacement,double velocity) const override
    {
        double volume = volume_0 + displacement * area;
        const double max_volume = volume_0 + max_length * area;
        const double min_volume = volume_0 + min_length * area;
        volume = std::clamp(volume,min_volume,max_volume);
        double volume_dot = area * velocity;
        return {volume, volume_dot};
    }
    std::vector<double> calculate_3_elements(const Eigen::Vector<double,4>& States) const override
    {
        std::vector<double> elements(3);
        elements[0] = -0.1 * 1e3; // K = N/m = 1e-3 M/mm
        if (States[1] > 0)
        {
            elements[1] = -0.02 * 1e3; // B = N/m/s = 1e-3 Ns/mm
        }
        else {elements[1] = -0.01 * 1e3;}
        elements[2] =  area;
        return elements;
    }
};

#endif //LIB_SIM_H