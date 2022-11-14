using DelimitedFiles
using Plots

include("functions//circuit.jl")
include("parameters.jl")

# want to vary two types of variables
# vary the order and block size
# vary the amplitude or the bias

# initialise the plots
samples_1 = []
samples_2 = []
fid_list = []
gate_fid_list = []

ave_fid = zeros(Float64, length(x))
ave_fid_SE = zeros(Float64, length(x))

ave_gate_fid = zeros(Float64, length(x))
ave_gate_SE = zeros(Float64, length(x))

ARGS = ["opt", "test"]
# now we save onto a folder
#open("parameters.txt", "w") do file
open(string("data_", ARGS[2], "/parameters_", ARGS[1],".txt"), "w") do file
    write(file, string("code: ", code, "\n"))
    write(file, string("sample_no: ", sample_no, "\n"))
    write(file, string("n: ", block_size[1], "\n"))
    write(file, string("m: ", block_size[2], "\n"))
    write(file, string("rep: ", block_size[3], "\n"))
    write(file, string("N_ord: ", N_ord, "\n"))
    write(file, "-------------------------------------\n")
    if x_var == "gamma"
        write(file, string("loss_1: ", x_min, "-", x_max, ", x_num ", x_num, "\n"))
        write(file, string("loss_2: ", x_min, "-", x_max, ", x_num ", x_num, "\n"))

        write(file, string("alpha: ", alpha, "\n"))
    else
        write(file, string("loss_1: ", err_place[1], ", ", err_info[1], "\n"))
        write(file, string("loss_2: ", err_place[3], ", ", err_info[3], "\n"))
    end
    write(file, string("dephase_1: ", err_place[2], ", ", err_info[2], "\n"))
    write(file, string("dephase_2: ", err_place[4], ", ", err_info[4], "\n"))
    write(file, "-------------------------------------\n")
    write(file, string("measurement_type: ", measure, "\n"))
    write(file, string("decode_type: ", decode_type, "\n"))
    write(file, string("error_spread_type: ", err_spread_type, "\n"))
    write(file, "-------------------------------------\n")
    write(file, string("x_var: ", x_var, "\n"))
    if x_var == "alpha"
        write(file, string("alpha: ", x_min, "-", x_step, "-", x_max, "\n"))
        write(file, string("bias: ", bias, "\n"))
        if dif_alpha
            write(file, string("differ_alpha: ", dif_alpha, ", ", alpha_2, "\n"))
        end
    elseif x_var == "bias"
        write(file, string("bias: ", x_min, "-", x_step, "-", x_max, "\n"))
        write(file, string("alpha: ", alpha, "\n"))
    end
end

@time begin
    for i = 1:length(x)
        println("-------------------------------------")
        @time begin
            if x_var == "alpha"
                println("x_vary: ", x_var, " | ", "x: ", x[i], " | ancilla mode alpha : ",  dif_alpha, "-", alpha_2, " | order: ", N_ord, " | block size (row, col, rep): ", block_size, " | decoder: ", decode_type)
                if dif_alpha == false
                    ave_fid[i], ave_gate_fid[i], fid_list_temp, gate_fid_list_temp, samples_1_temp, samples_2_temp, ave_fid_SE[i], ave_gate_SE[i] = circuit(code, N_ord, [x[i], x[i]], block_size, err_place, err_info, measure, decode_type, sample_no, bias, err_spread_type)
                else
                    ave_fid[i], ave_gate_fid[i], fid_list_temp, gate_fid_list_temp, samples_1_temp, samples_2_temp, ave_fid_SE[i], ave_gate_SE[i] = circuit(code, N_ord, [x[i], alpha_2], block_size, err_place, err_info, measure, decode_type, sample_no, bias, err_spread_type)
                end
            elseif x_var == "gamma"
                println("x_vary: ", x_var, " | x: ", x[i], " | alpha: ", alpha[1], ", ", alpha[2], " | order: ", N_ord, " | block size (row, col, rep): ", block_size, " | decoder: ", decode_type)

                ave_fid[i], ave_gate_fid[i], fid_list_temp, gate_fid_list_temp, samples_1_temp, samples_2_temp, ave_fid_SE[i], ave_gate_SE[i] = circuit(code, N_ord, alpha, block_size, err_place, [x[i], 0.0, x[i], 0.0], measure, decode_type, sample_no, [0, 0], err_spread_type)
            end
        end

        append!(samples_1, samples_1_temp)
        append!(samples_2, samples_2_temp)
        push!(fid_list, fid_list_temp)
        push!(gate_fid_list, gate_fid_list_temp)
        
        # Write it as we go
        writedlm(string("data_", ARGS[2], "/", ARGS[1], ".csv"), ave_fid, ',')
        writedlm(string("data_", ARGS[2], "/fidelity_list_", ARGS[1], ".csv"), fid_list, ',')
        writedlm(string("data_", ARGS[2], "/SE_", ARGS[1], ".csv"), ave_fid_SE, ',')

        writedlm(string("data_", ARGS[2], "/gate_", ARGS[1], ".csv"), ave_gate_fid, ',')
        writedlm(string("data_", ARGS[2], "/gate_fidelity_list_", ARGS[1], ".csv"), gate_fid_list, ',')
        writedlm(string("data_", ARGS[2], "/gate_SE_", ARGS[1], ".csv"), ave_gate_SE, ',')
    end
end


###########################################
#                                         #
#                PLOTTING                 #
#                                         #
###########################################

#plot(x, 1 .- ave_fid)

#spot = 1
#p1_plus, p1_min = samples_plot(samples_1[sample_no*(spot - 1) + 1:sample_no*spot], N_ord[1], x[spot], measure[1], bias[1])
#p2_plus, p2_min = samples_plot(samples_2[sample_no*(spot - 1) + 1:sample_no*spot], N_ord[2], x[spot], measure[2], bias[2])

#p_fid = initial_plotting(ave_fid, ave_fid_SE, x)
#p_gate = initial_plotting(ave_gate_fid, ave_gate_SE, x)

#savefig(p_fid, string("data_", ARGS[2], "/plot_fid_", ARGS[1]))
#savefig(p_gate, string("data_", ARGS[2], "/plot_gate_fid_", ARGS[1]))
#savefig(p1_plus, "samples_1_plot_plus")
#savefig(p1_min, "samples_1_plot_min")
#savefig(p2_plus, "samples_2_plot_plus")
#savefig(p2_min, "samples_2_plot_min")
