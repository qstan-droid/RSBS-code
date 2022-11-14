using QuantumOptics
using DelimitedFiles

# use a bunch of shit
include("functions/code_prep.jl")
include("functions/errors.jl")
include("functions/measurement.jl")
include("functions/decode.jl")
include("functions/fidelity.jl")

#-----------------------------------------#
# starting variables
sample_size = 1

code = ["binomial", "binomial"]
meas_type = ["opt_phase", "opt_phase"]
dcode_type = "ml_ave"

N_ord = [2, 2]
alpha_1 = 6:1:10
alpha_2 = 15

# error stuff
nu = [0.01, 0.01]

ave_fid = zeros(length(alpha_1))
ave_err = zeros(length(alpha_1))

ave_fid_gate = zeros(length(alpha_1))
ave_err_gate = zeros(length(alpha_1))

fid_list = []
gate_fid_list = []

@time begin
    for k = 1:length(alpha_1)

        @time begin
            #-----------------------------------------#
            # find cut-off dim and prepare codestates
            dim_1 = find_dim(code[1], alpha_1[k], N_ord[1])
            dim_2 = find_dim(code[2], alpha_1[k], N_ord[2])

            xbasis_1 = code_prep(N_ord[1], dim_1, alpha_1[k], code[1])
            xbasis_2 = code_prep(N_ord[2], dim_2, alpha_1[k], code[2])

            # sample errors
            l1, loss_norm_1 = loss_sample(true, nu[1], 1, 1, xbasis_1, sample_size)
            l2, loss_norm_2 = loss_sample(true, nu[2], 1, 1, xbasis_2, sample_size)

            # make the error transparent unitary
            b1 = xbasis_1[7]
            b2 = xbasis_2[7]

            dn = min(N_ord[1], N_ord[2])
            gt = pi # radians of rotation

            # upper bounds for U_cr
            up_bound_1 = Int(floor(alpha_1[k]/2) - 1)
            up_bound_2 = Int(floor(alpha_1[k]/2) - 1)

            if up_bound_1 <= 0
                up_bound_1 = 0
            end

            if up_bound_2 <= 0
                up_bound_2 = 0
            end

            # original Ucr
            #U_cr = tensor(identityoperator(b1), identityoperator(b2)) + (exp(1im*gt) - 1)*sum(sum(sum(tensor(projector(fockstate(b1, (2*j+1)*N_ord[1] - la)), projector(fockstate(b2, (2*j+1)*N_ord[2] - lb))) for lb = 0:dn-1-la) for la = 0:dn-1) for j = 0:up_bound_1)
            
            # "corrected" (lmao) Ucr
            U_cr = tensor(identityoperator(b1), identityoperator(b2)) + (exp(1im*gt) - 1)*sum(sum(sum(sum(tensor(projector(fockstate(b1, (2*j+1)*N_ord[1] - la)), projector(fockstate(b2, (2*k+1)*N_ord[2] - lb))) for lb = 0:dn-1-la) for la = 0:dn-1) for j = 0:up_bound_1) for k = 0:up_bound_2)
         
            #-----------------------------------------#
            # error terms
            E = function(x::Int64, nu, n_b, a_b) # error term for 1
                (((1 - exp(-nu))^(x/2))/(sqrt(factorial(big(x)))))*exp(-nu/2*dense(n_b))*(a_b^x)
            end

            F = function(x::Int64, nu, n_b, a_b, x_alt::Int64, N_ord_min) # error term 2
                exp(1im*floor(x_alt/N_ord_min)*pi*dense(n_b))*(((1 - exp(-nu))^(x/2))/(sqrt(factorial(big(x)))))*exp(-nu/2*dense(n_b))*(a_b^x)
            end

            E_eff = function(x::Int64, nu, n_b, a_b, x_alt::Int64, N_ord_min) # error term for 1
                (((1 - exp(-nu))^(x/2))/(sqrt(factorial(big(x)))))*exp(-nu/2*dense(n_b))*(a_b^x)*exp(1im*floor(x_alt/N_ord_min)*(pi/N_ord[1])*dense(n_b))
            end

            F_eff = function(x::Int64, nu, n_b, a_b, x_alt::Int64, N_ord_min) # error term 2
                exp(1im*floor(x_alt/N_ord_min)*(pi/N_ord[2])*dense(n_b))*(((1 - exp(-nu))^(x/2))/(sqrt(factorial(big(x)))))*exp(-nu/2*dense(n_b))*(a_b^x)
            end

            # initialise terms array
            terms = fill(U_cr*tensor(E(l1[1, 1, 1], nu[1], xbasis_1[3], xbasis_1[4]), E(l2[1, 1, 1], nu[2], xbasis_2[3], xbasis_2[4]))*dagger(U_cr)*tensor(xbasis_1[1], xbasis_2[1]), (sample_size, 4))
            
            for i = 1:sample_size
                err = U_cr*tensor(E(l1[1, 1, i], nu[1], xbasis_1[3], xbasis_1[4]), E(l2[1, 1, i], nu[2], xbasis_2[3], xbasis_2[4]))*dagger(U_cr)

                terms[i, 1] = err*tensor(xbasis_1[1], xbasis_2[1])
                terms[i, 2] = err*tensor(xbasis_1[1], xbasis_2[2])
                terms[i, 3] = err*tensor(xbasis_1[2], xbasis_2[1])
                terms[i, 4] = err*tensor(xbasis_1[2], xbasis_2[2])
            end

            #-----------------------------------------#
            # measurement operators
            meas1 = measurement_operator(meas_type[1], xbasis_1, N_ord[1], 0)
            meas2 = measurement_operator(meas_type[2], xbasis_2, N_ord[2], 0)

            # pdfs for the two modes
            function pdf_single_1(terms, meas_op, samp, b, loss_norm_1, loss_norm_2)
                #ans = (dagger(terms[1])*meas_op(samp)*terms[1] + dagger(terms[2])*meas_op(samp)*terms[2])*(dagger(terms[3])*terms[3] + dagger(terms[4])*terms[4])/(4*loss_norm_2*loss_norm_1)
                ans = (dagger(terms[1])*tensor(meas_op(samp), identityoperator(b))*terms[1] + dagger(terms[2])*tensor(meas_op(samp), identityoperator(b))*terms[2] + dagger(terms[3])*tensor(meas_op(samp), identityoperator(b))*terms[3] + dagger(terms[4])*tensor(meas_op(samp), identityoperator(b))*terms[4])/(4*loss_norm_1*loss_norm_2)
                return ans
            end

            function pdf_single_2(terms, x, x1, norms_1, meas_op_1, meas_op_2, loss_norm_1, loss_norm_2)
                #ans = (dagger(terms[3])*meas_op_2(x)*terms[3] + dagger(terms[4])*meas_op_2(x)*terms[4])*(dagger(terms[1])*meas_op_1(x1)*terms[1] + dagger(terms[2])*meas_op_1(x1)*terms[2])/(4*loss_norm_1*norms_1*loss_norm_2)
                ans = (dagger(terms[1])*tensor(meas_op_1(x1), meas_op_2(x))*terms[1] + dagger(terms[2])*tensor(meas_op_1(x1), meas_op_2(x))*terms[2] + dagger(terms[3])*tensor(meas_op_1(x1), meas_op_2(x))*terms[3] + dagger(terms[4])*tensor(meas_op_1(x1), meas_op_2(x))*terms[4])/(4*norms_1*loss_norm_1*loss_norm_2)
                return ans
            end

            #-----------------------------------------#
            # sampling

            # sampling for the first mode
            samp_1 = zeros(Complex{Float64}, (1, 1, sample_size))
            norms_1 = zeros(Complex{Float64}, sample_size)

            ceil = 1

            for i = 1:sample_size
                accept = false
                while accept == false
                    # get random sample
                    samp_1[1, 1, i] = sample_generator(code[1], meas_type[1], xbasis_1, N_ord, l1[1, 1, i], "no_spread", 1)
                    global f = abs(pdf_single_1(terms[i, :], meas1, samp_1[i], b2, loss_norm_1[i], loss_norm_2[i]))

                    # sample a random number
                    u = rand(Uniform(0, ceil))

                    if abs(f) > ceil
                        println("goes above one: ", abs(f))
                    else
                        if u < abs(f)
                            accept = true
                        end
                    end
                end

                norms_1[i] = f
            end

            println("first sampling done")

            # sampling for second mode
            samp_2 = zeros(Complex{Float64},(1, 1, sample_size))
            norms_2 = zeros(Complex{Float64},sample_size)

            for i = 1:sample_size
                accept = false

                while accept == false
                    # get random sample
                    samp_2[1, 1, i] = sample_generator(code[2], meas_type[2], xbasis_2, N_ord, l2[1, 1, i], "no_spread", 2)
                    global f = abs(pdf_single_2(terms[i, :], samp_2[i], samp_1[i], norms_1[i], meas1, meas2, loss_norm_1[i], loss_norm_2[i]))

                    # sample random number
                    u = rand(Uniform(0, ceil))

                    if abs(f) > ceil
                        println("goes above one: ", abs(f))
                    else
                        if u < abs(f)
                            accept = true
                        end
                    end
                end

                norms_2[i] = f
            end

            println("second sampling done")

            #-----------------------------------------#
            # find outcomes from samples

            if dcode_type == "ml_ave"
                outcome_1 = zeros(Bool, sample_size)
                outcome_2 = zeros(Bool, sample_size)

                l_max = findmin([xbasis_1[8]*N_ord[1], xbasis_2[8]*N_ord[2]])[1] + 5 # channel limit

                part = zeros(4)

                # preprepare the states
                err_state_1 = sum(sum(tensor(E(l1, nu[1], xbasis_1[3], xbasis_1[4]), E(l2, nu[2], xbasis_2[3], xbasis_2[4]))*dagger(U_cr)*tensor(dm(xbasis_1[1]), dm(xbasis_2[1]))*U_cr*tensor(dagger(E(l1, nu[1], xbasis_1[3], xbasis_1[4])), dagger(E(l2, nu[2], xbasis_2[3], xbasis_2[4]))) for l2 = 0:l_max) for l1 = 0:l_max)
                err_state_2 = sum(sum(tensor(E(l1, nu[1], xbasis_1[3], xbasis_1[4]), E(l2, nu[2], xbasis_2[3], xbasis_2[4]))*dagger(U_cr)*tensor(dm(xbasis_1[1]), dm(xbasis_2[2]))*U_cr*tensor(dagger(E(l1, nu[1], xbasis_1[3], xbasis_1[4])), dagger(E(l2, nu[2], xbasis_2[3], xbasis_2[4]))) for l2 = 0:l_max) for l1 = 0:l_max)
                err_state_3 = sum(sum(tensor(E(l1, nu[1], xbasis_1[3], xbasis_1[4]), E(l2, nu[2], xbasis_2[3], xbasis_2[4]))*dagger(U_cr)*tensor(dm(xbasis_1[2]), dm(xbasis_2[1]))*U_cr*tensor(dagger(E(l1, nu[1], xbasis_1[3], xbasis_1[4])), dagger(E(l2, nu[2], xbasis_2[3], xbasis_2[4]))) for l2 = 0:l_max) for l1 = 0:l_max)
                err_state_4 = sum(sum(tensor(E(l1, nu[1], xbasis_1[3], xbasis_1[4]), E(l2, nu[2], xbasis_2[3], xbasis_2[4]))*dagger(U_cr)*tensor(dm(xbasis_1[2]), dm(xbasis_2[2]))*U_cr*tensor(dagger(E(l1, nu[1], xbasis_1[3], xbasis_1[4])), dagger(E(l2, nu[2], xbasis_2[3], xbasis_2[4]))) for l2 = 0:l_max) for l1 = 0:l_max)

                N_1 = U_cr*err_state_1*dagger(U_cr)
                N_2 = U_cr*err_state_2*dagger(U_cr)
                N_3 = U_cr*err_state_3*dagger(U_cr)
                N_4 = U_cr*err_state_4*dagger(U_cr)

                for y = collect(1:sample_size)
                    meas_mul = tensor(meas1(samp_1[y]), meas2(samp_2[y]))

                    part[1] = norm(tr(meas_mul*N_1))
                    part[2] = norm(tr(meas_mul*N_2))
                    part[3] = norm(tr(meas_mul*N_3))
                    part[4] = norm(tr(meas_mul*N_4))

                    max_index = findmax(part)[2]

                    if max_index == 1
                        outcome_1[y] = true
                        outcome_2[y] = true
                    elseif max_index == 2
                        outcome_1[y] = true
                        outcome_2[y] = false
                    elseif max_index == 3
                        outcome_1[y] = false
                        outcome_2[y] = true
                    elseif max_index == 4
                        outcome_1[y] = false
                        outcome_2[y] = false
                    end
                end
            elseif dcode_type == "naive"
                outcome_1, outcome_2 = decoding(samp_1, samp_2, N_ord, [1, 1, 1], 1, [0.01, 0, 0.01, 0], sample_size, "naive", meas_type, [0, 0], [xbasis_1, xbasis_2], code[1], 0, "wow")
            end

            #-----------------------------------------#
            # get fidelity
            fid = zeros(Float64, (sample_size, 1))
            fid_gate = zeros(Float64, (sample_size, 1))

            # prepare spin basis
            spin_b = SpinBasis(1//2)
            zero = spinup(spin_b)
            one = spindown(spin_b)

            I_d = identityoperator(spin_b)
            X_d = sigmax(spin_b)
            Z_d = sigmaz(spin_b)

            psi_ini = (tensor(zero, zero) + tensor(one, one))/sqrt(2)

            d = dim_1*N_ord[1] + 1

            for i = 1:sample_size

                if meas_type[1] == "heterodyne"
                    meas_dagger_1 = dagger(coherentstate(b1, samp_1[i]))
                    meas_dagger_2 = dagger(coherentstate(b2, samp_2[i]))
                elseif meas_type[1] == "opt_phase"
                    meas_dagger_1 = dagger(sum(exp(1im*m*samp_1[i])*fockstate(xbasis_1[7], m) for m = 0:xbasis_1[8]*N_ord[1])/sqrt(xbasis_1[8]*N_ord[1] + 1))
                    meas_dagger_2 = dagger(sum(exp(1im*m*samp_2[i])*fockstate(xbasis_2[7], m) for m = 0:xbasis_2[8]*N_ord[2])/sqrt(xbasis_2[8]*N_ord[2] + 1))
                end

                out_coeff = zeros(Complex{Float64}, 4)

                out_coeff[1] = tensor(meas_dagger_1, meas_dagger_2)*terms[i, 1]
                out_coeff[2] = tensor(meas_dagger_1, meas_dagger_2)*terms[i, 2]
                out_coeff[3] = tensor(meas_dagger_1, meas_dagger_2)*terms[i, 3]
                out_coeff[4] = tensor(meas_dagger_1, meas_dagger_2)*terms[i, 4]


                # prepare the final outcomes
                psi_out = normalize((out_coeff[1]*psi_ini + out_coeff[2]*tensor(X_d, I_d)*psi_ini + out_coeff[3]*tensor(Z_d, I_d)*psi_ini + out_coeff[4]*tensor(X_d*Z_d, I_d)*psi_ini)/4)

                if outcome_1[i] == true && outcome_2[i] == true
                    correction = tensor(I_d, I_d)
                elseif outcome_1[i] == true && outcome_2[i] == false
                    correction = tensor(I_d, X_d)
                elseif outcome_1[i] == false && outcome_2[i] == true
                    correction = tensor(I_d, Z_d)
                elseif outcome_1[i] == false && outcome_2[i] == false
                    correction = tensor(I_d, Z_d*X_d)
                end

                psi_corr = correction*psi_out

                # record fidelity
                fid[i] = norm(dagger(psi_ini)*psi_corr*dagger(psi_corr)*psi_ini)
                fid_gate[i] = (d*fid[i] + 1)/(d + 1)
            end

            # store fidelity
            ave_fid[k] = sum(fid)/sample_size
            ave_err[k] = find_SE(ave_fid[k], fid)

            ave_fid_gate[k] = sum(fid_gate)/sample_size
            ave_err_gate[k] = find_SE(ave_fid_gate[k], fid_gate)

            push!(fid_list, fid)
            push!(gate_fid_list, fid_gate)

            println("finished ", k, " point with fidelity: ", ave_fid[k])
            println("finished ", k, " point with ave err: ", ave_err[k])
            println("-------")
            println("finished ", k, " point with gate fidelity: ", ave_fid_gate[k])
            println("finished ", k, " point with ave gate err: ", ave_err_gate[k])
        end
    end
end
ARGS = ["1", "data_1_3_1_gamma_same_err_no_diff_K4_N1"]

writedlm(string("data_", ARGS[2], "/", ARGS[1], ".csv"), ave_fid, ',')
writedlm(string("data_", ARGS[2], "/gate_", ARGS[1], ".csv"), ave_fid_gate, ',')

writedlm(string("data_", ARGS[2], "/fidelity_list_", ARGS[1], ".csv"), fid_list, ',')
writedlm(string("data_", ARGS[2], "/gate_fidelity_list_", ARGS[1], ".csv"), gate_fid_list, ',')

# write the simulation notes down
open(string("data_", ARGS[2], "/parameters_", ARGS[1], ".txt"), "w") do file
    write(file, string("code: ", code, "\n"))
    write(file, string("sample_no: ", sample_size, "\n"))
    write(file, string("N_ord: ", N_ord, "\n"))
    write(file, string("measurement_type: ", meas_type, "\n"))
    write(file, string("decode_type: ", dcode_type, "\n"))
    write(file, "-------------------------------------\n")
    write(file, string("loss_1: ", nu[1], "\n"))
    write(file, string("loss_2: ", nu[2], "\n"))
end

# remove these for actual simulation
writedlm(string("data_", ARGS[2], "/err_", ARGS[1], ".csv"), ave_err, ',')
writedlm(string("data_", ARGS[2], "/gate_err_", ARGS[1], ".csv"), ave_err_gate, ',')