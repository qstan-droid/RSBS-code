using JLD

include("code_prep.jl")
include("errors.jl")
include("measurement.jl")
include("decode.jl")
include("fidelity.jl")

# N_ord, dim and alpha are lists
function circuit(code, N_ord, alpha, block_size, err_place, err_info, measure, decode_type, sample_no, bias, err_spread_type)

    # find the dimensions based on alpha
    dim_1 = find_dim(code[1], alpha[1], N_ord[1])
    dim_2 = find_dim(code[2], alpha[2], N_ord[2])
    dim = [dim_1, dim_2]

    # begin code preparation
    # xbasis = [plus, min, n_b, a_b, zero, one, b, alpha, dim]
    xbasis_1 = code_prep(N_ord[1], dim[1], alpha[1], code[1])
    xbasis_2 = code_prep(N_ord[2], dim[2], alpha[2], code[2])
    xbasis = [xbasis_1, xbasis_2]

    # sample errors
    # block_size = [no_rows, no_cols, no_repetitions]
    # error_placement = [loss on block_1, dephase on block_1, loss on block_2, dephase on block_2]
    # error_info = [nu_loss_1, nu_dephase_1, nu_loss_2, nu_dephase_2]

    loss_sampling_style = true

    if loss_sampling_style
        println("sampling errors...")
        loss_1, loss_norm_1 = loss_sample(err_place[1], err_info[1], block_size[2], block_size[1], xbasis_1, sample_no)
        dephase_1, dephase_norm_1 = dephase_sample(err_place[2], err_info[2], block_size[2], block_size[1], sample_no)

        loss_2, loss_norm_2 = loss_sample(err_place[3], err_info[3], block_size[2]*block_size[3], block_size[1], xbasis_2, sample_no)
        dephase_2, dephase_norm_2 = dephase_sample(err_place[4], err_info[4], block_size[2]*block_size[3], block_size[1], sample_no)
    else
        loss_1, loss_norm_1 = new_loss_sample(err_place[1], err_info[1], block_size[2], block_size[1], xbasis_1, sample_no, block_size, 1)
        dephase_1, dephase_norm_1 = dephase_sample(err_place[2], err_info[2], block_size[2], block_size[1], sample_no)

        loss_2, loss_norm_2 = new_loss_sample(err_place[3], err_info[3], block_size[2]*block_size[3], block_size[1], xbasis_2, sample_no, block_size, 2)
        dephase_2, dephase_norm_2 = dephase_sample(err_place[4], err_info[4], block_size[2]*block_size[3], block_size[1], sample_no)
    end

    # propagate errors
    # we assume the brooks-preskill code
    println("propagating errors...")
    if err_spread_type == "normal" # only spread if we do spread_type "normal"
        loss_1, dephase_1, loss_2, dephase_2 = error_propagation(loss_1, dephase_1, loss_2, dephase_2, block_size, N_ord, sample_no)
    end

    # do measurements on both blocks
    # measure = [measure_type_1, measure_type_2]

    # First we pre-prepare the measurements to make it faster
    # first define the Kraus operator for error

    err_prep_1_zero, err_prep_1_one = error_prep(loss_1, dephase_1, err_info[1], xbasis_1, 1)
    err_prep_2_zero, err_prep_2_one = error_prep(loss_2, dephase_2, err_info[3], xbasis_2, 2)

    err_prep_1 = [err_prep_1_zero, err_prep_1_one]
    err_prep_2 = [err_prep_2_zero, err_prep_2_one]

    err_exp_1_zero, err_exp_1_one, err_exp_1_zo, err_exp_1_oz = error_exp(err_prep_1_zero, err_prep_1_one)
    err_exp_2_zero, err_exp_2_one, err_exp_2_zo, err_exp_2_oz = error_exp(err_prep_2_zero, err_prep_2_one)

    err_exp_1 = [err_exp_1_zero, err_exp_1_one, err_exp_1_zo, err_exp_1_oz]
    err_exp_2 = [err_exp_2_zero, err_exp_2_one, err_exp_2_zo, err_exp_2_oz]

    ###################################################################################################
    # if measurement is adaptive homodyne
    if measure[1] == "adapt_homo"
        H_mn = load("H_mn.jld", "H_mn")
    else 
        H_mn = nothing
    end

    # generate measurement samples
    samples_1 = 0
    norms_1 = 0     # this is just to fill in
    meas_exp_1 = 0

    println("sampling measurements...")
    @time begin
        samples_1, norms_1, meas_exp_plus_1, meas_exp_min_1, meas_exp_pm_1, meas_exp_mp_1, no_of_times_list_1 = measurement_samples(err_prep_1, err_prep_2, err_exp_1, err_exp_2, 1, measure, meas_exp_1, [xbasis_1, xbasis_2], N_ord, samples_1, norms_1, code, block_size, loss_norm_1, loss_norm_2, loss_1, loss_2, H_mn, alpha, err_info, err_spread_type)
        meas_exp_1 = [meas_exp_plus_1, meas_exp_min_1, meas_exp_pm_1, meas_exp_mp_1]
        println("second block sampling")
        samples_2, norms_2, meas_exp_zero_2, meas_exp_one_2, meas_exp_zo_2, meas_exp_oz_2, no_of_times_list_2 = measurement_samples(err_prep_1, err_prep_2, err_exp_1, err_exp_2, 2, measure, meas_exp_1, [xbasis_1, xbasis_2], N_ord, samples_1, norms_1, code, block_size, loss_norm_1, loss_norm_2, loss_1, loss_2, H_mn, alpha, err_info, err_spread_type)
        meas_exp_2 = [meas_exp_zero_2, meas_exp_one_2, meas_exp_zo_2, meas_exp_oz_2]
    end
    #println("acceptance probability 1: ", sum(no_of_times_list_1)/sample_no)
    #println("acceptance probability 2: ", sum(no_of_times_list_2)/sample_no)
    
    ###################################################################################################
    # Thus comes decoding
    # outcomes are the decoded outcomes for each block, should be an array of length(sample_no) for each
    # bias = [bias_amplitude_1, bias_amplitude_2]
    println("decoding...")
    @time begin
        outcomes_1, outcomes_2 = decoding(samples_1, samples_2, N_ord, block_size, err_place, err_info, sample_no, decode_type, measure, bias, xbasis, code, H_mn, err_spread_type)
    end
    
    ###################################################################################################
    # First we find the
    println("calculating fidelity...")
    P = find_coeff(block_size, samples_1, samples_2, xbasis_1, xbasis_2, err_prep_1, err_prep_2, measure, N_ord, H_mn)
    ave_fidelity, fid_list, SE, ave_gate_fid, gate_fid_list, fid_gate_SE = fid_ave_func(outcomes_1, outcomes_2, P, N_ord, alpha)

    return ave_fidelity, ave_gate_fid, fid_list, gate_fid_list, samples_1, samples_2, SE, fid_gate_SE
end