using QuantumOptics
include("measurement.jl")

function meas_operator_misc(meas_type, xbasis, N_ord)

    coh_space = FockBasis(xbasis[9])

    if meas_type == "heterodyne"
        meas = function(x)
            dagger(coherentstate(coh_space, x))
        end
    elseif meas_type == "opt_phase"
        # Define the measurement operators
        meas = function(x)
            #agger(sum(exp(1im*n*x)*fockstate(coh_space, n) for n = 0:xbasis[8]*N_ord)/(xbasis[8]*N_ord + 1))
            dagger(sum(exp(1im*n*x)*fockstate(coh_space, n) for n = 0:xbasis[8]*N_ord))/sqrt(2*pi)
        end
    elseif meas_type == "homodyne"
        
    end

    return meas
end

##################################################
# Finds the coefficients behind each term
function find_coeff(block_size, samples_1, samples_2, xbasis_1, xbasis_2, err_prep_1, err_prep_2, measure, N_ord, H_mn)

    row, col, sample_no = size(samples_1)
    rep = block_size[3]
    P = zeros(Complex{Float64}, (4, sample_no))

    if measure[1] != "adapt_homo"
        meas_1 = meas_operator_misc(measure[1], xbasis_1, N_ord[1])
        meas_2 = meas_operator_misc(measure[2], xbasis_2, N_ord[2])
    else
        meas_1 = measurement_operator(measure[1], xbasis_1, N_ord[1], H_mn)
        meas_2 = measurement_operator(measure[2], xbasis_2, N_ord[2], H_mn)
    end

    for k = 1:sample_no
        if measure[1] != "adapt_homo"
            # confirmed that abs is correct
            A_plus = (prod(prod(meas_1(samples_1[i, j, k])*err_prep_1[1][i, j, k] for j = 1:col) + prod(meas_1(samples_1[i, j, k])*err_prep_1[2][i, j, k] for j = 1:col) for i = 1:row) + prod(prod(meas_1(samples_1[i, j, k])*err_prep_1[1][i, j, k] for j = 1:col) - prod(meas_1(samples_1[i, j, k])*err_prep_1[2][i, j, k] for j = 1:col) for i = 1:row))/(sqrt(2)^(row+1))
            A_min = (prod(prod(meas_1(samples_1[i, j, k])*err_prep_1[1][i, j, k] for j = 1:col) + prod(meas_1(samples_1[i, j, k])*err_prep_1[2][i, j, k] for j = 1:col) for i = 1:row) - prod(prod(meas_1(samples_1[i, j, k])*err_prep_1[1][i, j, k] for j = 1:col) - prod(meas_1(samples_1[i, j, k])*err_prep_1[2][i, j, k] for j = 1:col) for i = 1:row))/(sqrt(2)^(row+1))

            B_plus = prod(prod(prod(meas_2(samples_2[i, (p-1)*col + j, k])*err_prep_2[1][i, (p-1)*col + j, k] for j = 1:col) + prod(meas_2(samples_2[i, (p-1)*col + j, k])*err_prep_2[2][i, (p-1)*col + j, k] for j = 1:col) for p = 1:rep) for i = 1:row)/(sqrt(2)^(row*rep))
            B_min = prod(prod(prod(meas_2(samples_2[i, (p-1)*col + j, k])*err_prep_2[1][i, (p-1)*col + j, k] for j = 1:col) - prod(meas_2(samples_2[i, (p-1)*col + j, k])*err_prep_2[2][i, (p-1)*col + j, k] for j = 1:col) for p = 1:rep) for i = 1:row)/(sqrt(2)^(row*rep))
        else
            #A_plus = sqrt(sum(sum(dagger(meas_1(samples_1[i, j, k])*err_prep_1[1][i, j, k])*meas_1(samples_1[i, j, k])*err_prep_1[1][i, j, k] for j = 1:col) + sum(dagger(meas_1(samples_1[i, j, k])*err_prep_1[1][i, j, k])*meas_1(samples_1[i, j, k])*err_prep_1[2][i, j, k] for j = 1:col) + sum(dagger(meas_1(samples_1[i, j, k])*err_prep_1[2][i, j, k])*meas_1(samples_1[i, j, k])*err_prep_1[1][i, j, k] for j = 1:col) + sum(dagger(meas_1(samples_1[i, j, k])*err_prep_1[2][i, j, k])*meas_1(samples_1[i, j, k])*err_prep_1[2][i, j, k] for j = 1:col) for i = 1:row)/2^row)
            #A_min = sqrt(sum(sum(dagger(meas_1(samples_1[i, j, k])*err_prep_1[1][i, j, k])*meas_1(samples_1[i, j, k])*err_prep_1[1][i, j, k] for j = 1:col) - sum(dagger(meas_1(samples_1[i, j, k])*err_prep_1[1][i, j, k])*meas_1(samples_1[i, j, k])*err_prep_1[2][i, j, k] for j = 1:col) - sum(dagger(meas_1(samples_1[i, j, k])*err_prep_1[2][i, j, k])*meas_1(samples_1[i, j, k])*err_prep_1[1][i, j, k] for j = 1:col) + sum(dagger(meas_1(samples_1[i, j, k])*err_prep_1[2][i, j, k])*meas_1(samples_1[i, j, k])*err_prep_1[2][i, j, k] for j = 1:col) for i = 1:row)/2^row)

            A_plus = sqrt(sum(sum(dagger(meas_1(samples_1[i, j, k])*err_prep_1[1][i, j, k])*meas_1(samples_1[i, j, k])*err_prep_1[1][i, j, k] for j = 1:col) + sum(dagger(meas_1(samples_1[i, j, k])*err_prep_1[1][i, j, k])*meas_1(samples_1[i, j, k])*err_prep_1[2][i, j, k] for j = 1:col) + sum(dagger(meas_1(samples_1[i, j, k])*err_prep_1[2][i, j, k])*meas_1(samples_1[i, j, k])*err_prep_1[1][i, j, k] for j = 1:col) + sum(dagger(meas_1(samples_1[i, j, k])*err_prep_1[2][i, j, k])*meas_1(samples_1[i, j, k])*err_prep_1[2][i, j, k] for j = 1:col) for i = 1:row)/2^row)
            A_min = sqrt(sum(sum(dagger(meas_1(samples_1[i, j, k])*err_prep_1[1][i, j, k])*meas_1(samples_1[i, j, k])*err_prep_1[1][i, j, k] for j = 1:col) - sum(dagger(meas_1(samples_1[i, j, k])*err_prep_1[1][i, j, k])*meas_1(samples_1[i, j, k])*err_prep_1[2][i, j, k] for j = 1:col) - sum(dagger(meas_1(samples_1[i, j, k])*err_prep_1[2][i, j, k])*meas_1(samples_1[i, j, k])*err_prep_1[1][i, j, k] for j = 1:col) + sum(dagger(meas_1(samples_1[i, j, k])*err_prep_1[2][i, j, k])*meas_1(samples_1[i, j, k])*err_prep_1[2][i, j, k] for j = 1:col) for i = 1:row)/2^row)

            B_plus = sqrt(sum(sum(sum(dagger(meas_2(samples_2[i, (p-1)*col + j, k])*err_prep_2[1][i, (p-1)*col + j, k])*meas_2(samples_2[i, (p-1)*col + j, k])*err_prep_2[1][i, (p-1)*col + j, k] for j = 1:col) + sum(dagger(meas_2(samples_2[i, (p-1)*col + j, k])*err_prep_2[1][i, (p-1)*col + j, k])*meas_2(samples_2[i, (p-1)*col + j, k])*err_prep_2[2][i, (p-1)*col + j, k] for j = 1:col) + sum(dagger(meas_2(samples_2[i, (p-1)*col + j, k])*err_prep_2[2][i, (p-1)*col + j, k])*meas_2(samples_2[i, (p-1)*col + j, k])*err_prep_2[1][i, (p-1)*col + j, k] for j = 1:col) + sum(dagger(meas_2(samples_2[i, (p-1)*col + j, k])*err_prep_2[2][i, (p-1)*col + j, k])*meas_2(samples_2[i, (p-1)*col + j, k])*err_prep_2[2][i, (p-1)*col + j, k] for j = 1:col) for p = 1:rep) for i = 1:row)/2^(row*rep))
            B_min = sqrt(sum(sum(sum(dagger(meas_2(samples_2[i, (p-1)*col + j, k])*err_prep_2[1][i, (p-1)*col + j, k])*meas_2(samples_2[i, (p-1)*col + j, k])*err_prep_2[1][i, (p-1)*col + j, k] for j = 1:col) - sum(dagger(meas_2(samples_2[i, (p-1)*col + j, k])*err_prep_2[1][i, (p-1)*col + j, k])*meas_2(samples_2[i, (p-1)*col + j, k])*err_prep_2[2][i, (p-1)*col + j, k] for j = 1:col) - sum(dagger(meas_2(samples_2[i, (p-1)*col + j, k])*err_prep_2[2][i, (p-1)*col + j, k])*meas_2(samples_2[i, (p-1)*col + j, k])*err_prep_2[1][i, (p-1)*col + j, k] for j = 1:col) + sum(dagger(meas_2(samples_2[i, (p-1)*col + j, k])*err_prep_2[2][i, (p-1)*col + j, k])*meas_2(samples_2[i, (p-1)*col + j, k])*err_prep_2[2][i, (p-1)*col + j, k] for j = 1:col) for p = 1:rep) for i = 1:row)/2^(row*rep))
        end

        P[1, k] = A_plus*B_plus
        P[2, k] = A_plus*B_min
        P[3, k] = A_min*B_plus
        P[4, k] = A_min*B_min
    end

    return P
end

##################################################
# Finds the average fidelity

function fid_ave_func(outcomes_1, outcomes_2, P, N_ord, x)

    # initialise the state
    b = SpinBasis(1//2)
    I_d = identityoperator(b)
    X_d = sigmax(b)
    Z_d = sigmaz(b)

    zero = spinup(b)
    one = spindown(b)

    psi_ini = (tensor(zero, zero) + tensor(one, one))/sqrt(2)

    sample_no = length(outcomes_1)
    fid_list = zeros(Float64, (sample_no, 1))
    fid_gate_list = zeros(Float64, (sample_no, 1))

    # calculating gate fidelity
    N_ord_1 = N_ord[1]
    x_1 = x[1] 

    d = x_1*N_ord_1 + 1

    # outcomes print out

    for k = 1:sample_no
        psi_out = normalize((P[1, k]*psi_ini + P[2, k]*tensor(I_d, X_d)*psi_ini + P[3, k]*tensor(I_d, Z_d)*psi_ini + P[4, k]*tensor(I_d, X_d*Z_d)*psi_ini)/2)

        correction_no = 1

        if outcomes_1[k] == true && outcomes_2[k] == true
            correction = tensor(I_d, I_d)
            correction_no = 1
            #println(correction_no)
        elseif outcomes_1[k] == true && outcomes_2[k] == false
            correction = tensor(I_d, X_d)
            correction_no = 2
            #println(correction_no)
        elseif outcomes_1[k] == false && outcomes_2[k] == true
            correction = tensor(I_d, Z_d)
            correction_no = 3
            #println(correction_no)
        elseif outcomes_1[k] == false && outcomes_2[k] == false
            correction = tensor(I_d, Z_d*X_d)
            correction_no = 4
            #println(correction_no)
        end

        # note 
        # correction_no = 1 ~ I
        # correction_no = 2 ~ X
        # correction_no = 3 ~ Z
        # correction_no = 4 ~ XZ
        
        psi_corr = correction*psi_out

        # record fidelity
        #fid_list[k] = real(fidelity(psi_corr_dm, psi_ini_dm))^2
        fid_list[k] = norm(dagger(psi_ini)*psi_corr*dagger(psi_corr)*psi_ini)
        fid_gate_list[k] = (d*fid_list[k] + 1)/(d + 1)

        #println("gives a fidelity of: ", fid_list[k])

    end

    # calculate average fidelity
    fid_ave = sum(fid_list)/sample_no
    fid_gate_ave = sum(fid_gate_list)/sample_no
    
    # find the standard error
    fid_SE = find_SE(fid_ave, fid_list)
    fid_gate_SE = find_SE(fid_gate_ave, fid_gate_list)
    
    println("average fid: ", fid_ave)
    println("standard error: ", fid_SE)
    println("---------------------------")
    println("average gate fid: ", fid_gate_ave)
    println("standard error gate: ", fid_gate_SE)
    return fid_ave, fid_list, fid_SE, fid_gate_ave, fid_gate_list, fid_gate_SE
end

##################################################
################ STATISTICS TOOLS ################
##################################################

function find_SE(average, data)

    n_size = length(data)

    # find standard deviation
    #sd = sqrt(sum((data[j] - average)^2 for j = 1:n_size)/(n_size-1))
    sd = sqrt(average*(1-average))

    return sd/sqrt(n_size)
end