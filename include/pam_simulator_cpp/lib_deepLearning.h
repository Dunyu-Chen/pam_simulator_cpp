#ifndef LIB_DEEPLEARNING_H
#define LIB_DEEPLEARNING_H

#include "torch/torch.h"

class Dynamic_RBF_Net : public torch::nn::Module
{
public:
    int input_size,hidden_size,output_size;
    torch::nn::LSTM rbf_params_gen;
    torch::nn::Linear output_mapper;
    Dynamic_RBF_Net(int Input_size,int Hidden_size, int Output_size) :
        input_size(Input_size), hidden_size(Hidden_size), output_size(Output_size),
        rbf_params_gen(
            register_module(
                "rbf_params_gen",
                torch::nn::LSTM(
                    torch::nn::LSTMOptions(input_size,(input_size+1)*hidden_size)// i x h centers ; h spreads
                    .batch_first(true).num_layers(1)
                )
            )
        ),
        output_mapper(
            register_module(
                "output_mapper",
                torch::nn::Linear(
                    torch::nn::LinearOptions(hidden_size,output_size)
                )
            )
        )
    {}

    torch::Tensor forward(torch::Tensor Inputs)
    {
        auto [raw_rbf_params,rbf_params_gen_hidden_states] = rbf_params_gen->forward(Inputs); // {b,s,(i+1)*h}
        auto flattened_rbf_params = raw_rbf_params.view({Inputs.size(0),Inputs.size(1),hidden_size,input_size+1}); // {b,s,h,i} 展开参数方便取值与广播
        auto centers = flattened_rbf_params.slice(-1,0,-1,1); // {b,s,h,i} 取最后一维前 i 个元素构建rbf centers
        auto spreads =
            torch::clamp(
                torch::softplus(flattened_rbf_params.slice(-1,-1,INT64_MAX,1).squeeze(-1))
                ,1e-3,5.0
            ); // {b,s,h} 取最后一维最后一个元素构建 spreads 限制其范围
        auto distance = torch::norm(Inputs.unsqueeze(-2)-centers,2,-1); //{b,s,1,i} - {b,s,h,i} -> {b,s,h,i} ->norm^2-> {b,s,h}
        auto rbf_outputs = torch::exp(
            torch::clamp(-spreads*distance.pow(2),-50.0,50.0)
        ); // {b,s,h}*{b,s,h} -> {b,s,h}
        auto outputs = torch::tanh(output_mapper->forward(rbf_outputs));// {b,s,h} -> {b,s,o}
        return outputs;
    }
};

class Dynamic_LSTM_RBF_Net : public torch::nn::Module
{
public:
    uint input_size,hidden_size,output_size;
    torch::nn::LSTM rbf_params_gen;
    torch::nn::LSTM output_mapper;
    torch::Tensor output_scale;
    torch::Tensor output_bias;
    torch::Tensor centers_mean;
    torch::Tensor centers_scale;
    Dynamic_LSTM_RBF_Net(uint Input_size, uint Hidden_size, uint Output_size) :
        input_size(Input_size),hidden_size(Hidden_size),output_size(Output_size),
        rbf_params_gen(
            register_module("rbf_params_gen",
                torch::nn::LSTM(
                    torch::nn::LSTMOptions(input_size,hidden_size*(input_size+1))
                        .batch_first(true)
                        .num_layers(1)
                )
            )
        ),
        output_mapper(
            register_module("output_mapper",
                torch::nn::LSTM(
                    torch::nn::LSTMOptions(hidden_size,output_size)
                    .batch_first(true)
                    .num_layers(1)
                )
            )
        ),
        output_scale(
            register_buffer("output_scale",torch::ones(output_size))
        ),
        output_bias(
            register_buffer("output_bias",torch::zeros(output_size))
        ),
        centers_mean(
            register_parameter("centers_mean",torch::zeros(input_size))
        ),
        centers_scale(
            register_parameter("centers_scale",torch::ones(input_size))
        )
    {}

    void set_output_range(torch::Tensor Output_scales,torch::Tensor Output_means)
    {
        TORCH_CHECK(Output_means.dim() == 1,"Wrong output dimension");
        TORCH_CHECK(Output_scales.dim() == 1,"Wrong output dimension");
        TORCH_CHECK(Output_means.size(0) ==output_size,"Wrong output size");
        TORCH_CHECK(Output_scales.size(0) ==output_size,"Wrong output size");
        this->output_scale.set_data(Output_scales);
        this->output_bias.set_data(Output_means);
    }
    void calibrate_centers (torch::Tensor Input_data_samples)
    {   TORCH_CHECK(Input_data_samples.dim() == 2,"Wrong input data size"); // {s,i}
        TORCH_CHECK(Input_data_samples.size(-1) == input_size,"Wrong size of Input_data_samples");
        TORCH_CHECK(Input_data_samples.size(0) >= 10,"mast have over 10 samples to calibrate");
        auto mean_device = centers_mean.device();
        auto scale_device = centers_scale.device();
        auto data_mean = torch::mean(Input_data_samples, 0).to(mean_device); //{i}
        auto data_std = torch::std(Input_data_samples, 0, false).clamp_min(1e-5).to(scale_device); // {i}
        {
            torch::NoGradGuard no_grad;
            this->centers_mean.set_data(data_mean);
            this->centers_scale.set_data(data_std);
        }

    }

    torch::Tensor forward(torch::Tensor Inputs)
    {
        TORCH_CHECK(Inputs.size(-1) == input_size);
        auto [raw_rbf_params,rbf_gen_hidden_states] = rbf_params_gen->forward(Inputs); // {b,s,(i+1)*h}
        auto expended_params = raw_rbf_params.view({Inputs.size(0),Inputs.size(1),hidden_size,input_size+1}); // {b,s,h,i+1}
        //std::cout<<expended_params<<std::endl;
        auto raw_centers = expended_params.slice(-1,0,-1,1);// {b,s,h,i}
        auto centers = centers_mean.reshape({1,1,1,-1}) + centers_scale.reshape({1,1,1,-1}) * torch::tanh(raw_centers); //{1,1,1,i} * {b,s,h,i}
        //std::cout<<centers<<std::endl;
        auto spreads = torch::clamp(
            torch::softplus(
                expended_params.slice(-1,-1,INT64_MAX,1).squeeze(-1)
            ),
            1e-3,5.0); // {b,s,h}
        //std::cout<<spreads<<std::endl;
        auto distance = torch::norm(Inputs.unsqueeze(-2) - centers,2,-1); //{b,s,1,i} - {b,s,h,i} -> {b,s,h,i} ->norm^2-1-> {b,s,h}
        auto rbf_outputs = torch::exp(
            torch::clamp(
                -spreads * distance.pow(2),
                -50.0,50.0
            )
        );
        auto [outputs,outputs_hidden_states] = output_mapper->forward(rbf_outputs); // {b,s,o}
        return output_bias+output_scale*torch::tanh(outputs);
    }
};

class Dynamic_Mapped_RBF_Net : public torch::nn::Module
{
public:
    uint input_size,hidden_size,rbf_size,output_size;
    torch::nn::LSTM lstm_rbf_params;
    torch::nn::Linear centers_map;
    torch::nn::Linear spreads_map;
    torch::nn::Linear output_layer;
    torch::Tensor center_scale;
    Dynamic_Mapped_RBF_Net(uint Input_size, uint Hidden_size,uint Rbf_size, uint Output_size) :
        input_size(Input_size),hidden_size(Hidden_size),rbf_size(Rbf_size),output_size(Output_size),
        lstm_rbf_params(
            register_module("lstm_params",
                            torch::nn::LSTM(
                                torch::nn::LSTMOptions(input_size,hidden_size)
                                .batch_first(true)
                                .num_layers(1)
                            )
            )
        ),
        centers_map(
            register_module("centers_map",
                torch::nn::Linear(
                    torch::nn::LinearOptions(hidden_size,rbf_size*input_size)
                    )
            )
        ),
        spreads_map(
            register_module("spreads_map",
                torch::nn::Linear(
                    torch::nn::LinearOptions(hidden_size,rbf_size)
                )
            )
        ),
        output_layer(
            register_module("output_layer",
                torch::nn::Linear(
                    torch::nn::LinearOptions(rbf_size,output_size)
                )
            )
        ),
        center_scale(
            register_buffer("data_scale",
                torch::ones(input_size)
            )
        )
    {}

    torch::Tensor forward(torch::Tensor Inputs)
    {
        TORCH_CHECK(Inputs.size(-1) == input_size);
        auto [rbf_params,_] = lstm_rbf_params(Inputs);
        auto raw_centers = centers_map->forward(rbf_params);
        auto centers = center_scale*torch::tanh(raw_centers).view({Inputs.size(0),Inputs.size(1),rbf_size,input_size});
        auto raw_spreads = spreads_map->forward(rbf_params);
        auto spreads = torch::softplus(raw_spreads) + 1e-10;
        auto expanded_inputs = Inputs.unsqueeze(-2);
        auto distances = torch::norm(expanded_inputs-centers,2,-1); //{b,s,1,i} - {b,s,c,i}
        auto rbf = torch::exp(-spreads*distances.pow(2));
        auto mapped_rbf = output_layer->forward(rbf);
        return mapped_rbf;
    }
};

#endif //LIB_DEEPLEARNING_H
