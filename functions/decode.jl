using QuantumOptics
using Distributions
using BenchmarkTools
using Combinatorics

include("measurement.jl")
include("errors.jl")

function decoding(samples_1, samples_2, N_ord, block_size, err_place, err_info, sample_no, decode_type, measure, bias, xbasis, code, H_mn, err_spread_type)
    # initialise empty arrays
    outcomes_1 = zeros(Bool, sample_no)
    outcomes_2 = zeros(Bool, sample_no)

    # how to decode?
    if decode_type == "naive"
        # decode each qubit individually
        samples_out_1 = naive_decode(samples_1, N_ord, block_size, measure[1], bias[1], decode_type, err_info, 1, xbasis)
        samples_out_2 = naive_decode(samples_2, N_ord, block_size, measure[2], bias[2], decode_type, err_info, 2, xbasis)

        # decide outcome through these
        outcomes_1 = block_decode(samples_out_1, 1, block_size)
        outcomes_2 = block_decode(samples_out_2, 2, block_size)
    elseif decode_type == "ml_ave"
        # find the outcomes 
        meas_decode_1 = ml_ave(samples_1, N_ord, err_info, xbasis, measure, 1, block_size, H_mn, err_spread_type)
        meas_decode_2 = ml_ave(samples_2, N_ord, err_info, xbasis, measure, 2, block_size, H_mn, err_spread_type)

        # block_decode the outcomes
        outcomes_1 = block_decode(meas_decode_1, 1, block_size)
        outcomes_2 = block_decode(meas_decode_2, 2, block_size)

    end

    return outcomes_1, outcomes_2
end

#########################################################
# different functions, different decoding types

function naive_decode(samples, N_ord, block_size, measure, bias, decode_type, err_info, block_no, xbasis)
    # decide the outcome for each qubit individually
    row, col, sample_no = size(samples)
    samples_out = zeros(Int64, (row, col, sample_no))
    if decode_type == "naive_ave_bias"
        # bias is -phi as we want to add the bias angle to whatever result we get
        if block_no == 1
            bias = -find_ave_angle(err_info[3], N_ord, xbasis[2])
        elseif block_no == 2
            bias = -find_ave_angle(err_info[1], N_ord, xbasis[1])
        end
    end

    for k = 1:sample_no
        for i = 1:row
            for j = 1:col
                samples_out[i, j, k] = meas_outcome(samples[i, j, k], N_ord[block_no], measure, bias)
            end
        end
    end

    return samples_out
end

#########################################################
# maximum likelihood decoder

function ml_ave(samples, N_ord, err_info, xbasis, measure, block_no, block_size, H_mn, err_spread_type)
    row, col, sample_no = size(samples)
    outcomes = zeros(Int64, (row, col, sample_no))

    # prepare the vector for the likelihoods
    parts = zeros(Complex{Float64}, 2)

    # prepare the xbasis stuff
    xbasis_1 = xbasis[1]
    xbasis_2 = xbasis[2]

    # prepare the N_ords
    N_ord_1 = N_ord[1]
    N_ord_2 = N_ord[2]

    # prepare the rotation operator which error rate are we interested in
    rep = block_size[3]
    if err_spread_type == "normal"
        if block_no == 1  # interested in block 2 loss rate
            C = function(k)
                exp(-1im*k*rep*pi*dense(xbasis_1[3])/(N_ord_1*N_ord_2))
            end
            nu_l = err_info[3]
            xbasis_l = xbasis[2]
            
        elseif block_no == 2  # interested in block 1 loss rate
            C = function(k)
                exp(-1im*k*pi*dense(xbasis_2[3])/(N_ord_1*N_ord_2))
            end
            nu_l = err_info[1]
            xbasis_l = xbasis[1]
        end
    elseif err_spread_type == "no_spread"
        if block_no == 1
            nu_l = err_info[1]
            xbasis_l = xbasis[1]
        elseif block_no == 2
            nu_l = err_info[3]
            xbasis_l = xbasis[2]
        end

        C = function(k)
            (((1 - exp(-nu_l))^(k/2))/sqrt(factorial(big(k))))*exp(-nu_l*dense(xbasis_l[3])/2)*xbasis_l[4]^k

        end
    end
    # find the maximum l_max
    l_max = findmin([xbasis_1[8]*N_ord[1], xbasis_2[8]*N_ord[2]])[1] + 2

    # prepare measurement operator and plus and min states
    meas_op = measurement_operator(measure[block_no], xbasis[block_no], N_ord[block_no], H_mn)

    # prepare the states
    plus_err_state = [C(l)*xbasis[block_no][1] for l = 0:l_max]
    min_err_state = [C(l)*xbasis[block_no][2] for l = 0:l_max]


    for n = 1:sample_no
        for i = 1:row
            for j = 1:col
                if err_spread_type == "normal"
                    parts[1] = sum(abs(loss_pdf(nu_l, l, xbasis_l))*dagger(plus_err_state[l+1])*meas_op(samples[i, j, n])*plus_err_state[l+1] for l = 0:l_max)
                    parts[2] = sum(abs(loss_pdf(nu_l, l, xbasis_l))*dagger(min_err_state[l+1])*meas_op(samples[i, j, n])*min_err_state[l+1] for l = 0:l_max)
                elseif err_spread_type == "no_spread"
                    parts[1] = sum(dagger(plus_err_state[l+1])*meas_op(samples[i, j, n])*plus_err_state[l+1] for l = 0:l_max)
                    parts[2] = sum(dagger(min_err_state[l+1])*meas_op(samples[i, j, n])*min_err_state[l+1] for l = 0:l_max)
                end

                max_like_index = findmax(abs.(parts))[2]

                if max_like_index == 1
                    outcomes[i, j, n] = 1
                elseif max_like_index == 2
                    outcomes[i, j, n] = -1
                end
            end
        end
    end

    return outcomes
end

#########################################################
# functions which are tools for decoding

function meas_outcome(meas, N_ord, meas_type, bias)
    if meas_type == "heterodyne"
        phi = mod2pi(mod2pi(angle(meas)) + pi/(N_ord*2)) + bias
        if phi == 2*pi
            phi = 0
        end
        if phi > 2*pi
            phi = phi - 2*pi
        end
    elseif meas_type == "opt_phase"
        phi = mod2pi(mod2pi(convert(Float64, meas)) + pi/(N_ord*2)) + bias
        if phi == 2*pi
            phi = 0
        end
        if phi > 2*pi
            phi = phi - 2*pi
        end
    end

    k = 0
    edge = false

    while k < N_ord

        if phi > 0 + 2*k*pi/N_ord && phi < pi/N_ord + 2*k*pi/N_ord
            return +1
            edge = true
        elseif phi > pi/N_ord + 2*pi*k/N_ord && phi < (2*pi/N_ord) + 2*k*pi/N_ord
            return -1
            edge = true
        end
        k += 1
    end

    if edge == false
        coin = rand(1:2)
        if coin == 1
            return +1
        else
            return -1
        end
    end
end

function block_decode(samples_out, block_no, block_size)
    # unpack variables
    rep = block_size[3]
    row, col, sample_no = size(samples_out)
    maj = zeros(Bool, sample_no)

    if block_no == 1
        # we compute total parity of each column
        # take majority vote over all parities
        for k = 1:sample_no
            col_par = fill(1, col)

            for i = 1:col
                col_par[i] = prod(samples_out[j, i, k] for j = 1:row)
            end

            # take majority vote over all column parities
            no_pos = count(l->(l == 1), col_par)
            no_neg = count(l->(l == -1), col_par)

            if no_pos > no_neg
                maj[k] = true
            else
                maj[k] = false
            end
        end

    elseif block_no == 2
        # take parities of each row
        for k = 1:sample_no
            row_par = fill(1, (rep, row, sample_no))

            # take row parities
            for k = 1:sample_no
                for i = 1:row
                    for j = 1:rep
                        row_par[j, i, k] = prod(samples_out[i, l, k] for l = (j-1)*div(col, rep)+1:j*div(col, rep))
                    end
                end
            end
            # majority vote over repetitions
            row_par_real = fill(1, (row, sample_no))
            for k = 1:sample_no
                for i = 1:row
                    par_maj = count(l->(l==1), row_par[:, i, k])
                    if par_maj > rep/2
                        row_par_real[i, k] = 1
                    else
                        row_par_real[i, k] = -1
                    end
                end
            end

            # majority vote over all column parities
            maj = zeros(Bool, sample_no)
            for k = 1:sample_no
                row_maj = count(l->(l==1), row_par_real[:, k])
                if row_maj > row/2
                    maj[k] = true
                else
                    maj[k] = false
                end
            end
        end
    end
    return maj
end

#------------------------------------------------------------#
# getting list of permutations
function maj_vote(array)
    no_of_false = 0
    no_of_true = 0

    for i in 1:length(array)
        if array[i]
            no_of_true += 1
        else
            no_of_false += 1
        end
    end

    if no_of_true > no_of_false
        # println("return true")
        return true
    else
        # println("return false")
        return false
    end
end