using QuantumOptics
using Distributions

function measurement_operator(meas_type, xbasis, N_ord, H_mn)

    coh_space = FockBasis(xbasis[9])

    # Define the measurement operators
    if meas_type == "heterodyne"
        meas = function(x)
            tensor(coherentstate(coh_space, x), dagger(coherentstate(coh_space, x)))/pi
        end
    elseif meas_type == "opt_phase"
        meas = function(x)
            sum(sum(exp(1im*(m-n)*x)*tensor(fockstate(coh_space, m), dagger(fockstate(coh_space, n))) for n = 0:xbasis[8]*N_ord) for m = 0:xbasis[8]*N_ord)/(2*pi)
        end
    elseif meas_type == "adapt_homo"
        meas = function(x)
            sum(sum(exp(1im*x*(m-n))*H_mn[m+1, n+1]*tensor(fockstate(coh_space, m), dagger(fockstate(coh_space, n))) for n = 0:xbasis[8]*N_ord) for m = 0:xbasis[8]*N_ord)/(2*pi)
        end
    end

    return meas
end

# Finds expectation value of measurements
function meas_exp_prep(meas_op, sample, err_prep_plus, err_prep_min)
    meas_exp_plus_ret = dagger(err_prep_plus)*meas_op(sample)*err_prep_plus
    meas_exp_min_ret = dagger(err_prep_min)*meas_op(sample)*err_prep_min
    meas_exp_pm_ret = dagger(err_prep_plus)*meas_op(sample)*err_prep_min
    meas_exp_mp_ret = dagger(err_prep_min)*meas_op(sample)*err_prep_plus

    return meas_exp_plus_ret, meas_exp_min_ret, meas_exp_pm_ret, meas_exp_mp_ret
end

#######################
function measurement_samples(err_prep_1, err_prep_2, err_exp_1, err_exp_2, block_no, measure_type, meas_exp_1, xbasis, N_ord, samples_1, norms_1, code, block_size, loss_norm_1, loss_norm_2, loss_1, loss_2, H_mn, alpha, err_info, err_spread_type)

    #### Rejection sampling
    meas_op_1 = measurement_operator(measure_type[1], xbasis[1], N_ord[1], H_mn)
    meas_op_2 = measurement_operator(measure_type[2], xbasis[2], N_ord[2], H_mn)
    meas_ops = [meas_op_1, meas_op_2]

    # prepare the container for samples
    if block_no == 1
        row, col, sample_no = size(err_prep_1[1])
    elseif block_no == 2
        row, col, sample_no = size(err_prep_2[1])
    end

    samples = zeros(Complex{Float64}, row, col, sample_no)
    norms = fill(1.0+0.0*1im, row, col, sample_no)

    meas_exp_plus = zeros(Complex{Float64}, row, col, sample_no)
    meas_exp_min = zeros(Complex{Float64}, row, col, sample_no)
    meas_exp_pm = zeros(Complex{Float64}, row, col, sample_no)
    meas_exp_mp = zeros(Complex{Float64}, row, col, sample_no)

    meas_exp = [meas_exp_plus, meas_exp_min, meas_exp_pm, meas_exp_mp]

    # acceptance fraction
    no_of_times_list = zeros(Complex{Float64}, row, col, sample_no)

    for k = 1:sample_no
        for i = 1:row
            for j = 1:col
                samples[i, j, k], norms[i, j, k], meas_exp[1][i, j, k], meas_exp[2][i, j, k], meas_exp[3][i, j, k], meas_exp[4][i, j, k], no_of_times_list[i, j, k] = rejection_sampling(err_prep_1, err_prep_2, err_exp_1, err_exp_2, block_no, measure_type, xbasis, N_ord, samples, samples_1, code, block_size, meas_ops, meas_exp, [i, j, k], meas_exp_1, norms, norms_1, loss_norm_1[:, :, k], loss_norm_2[:, :, k], loss_1[:, :, k], loss_2[:, :, k], alpha, err_info, err_spread_type)
            end
        end
    end

    return samples, norms, meas_exp[1], meas_exp[2], meas_exp[3], meas_exp[4], no_of_times_list
end

function rejection_sampling(err_prep_1, err_prep_2, err_exp_1, err_exp_2, block_no, measure_type, xbasis, N_ord, samples, samples_1, code, block_size, meas_ops, meas_exp, loc, meas_exp_1, norms, norms_1, loss_norm_1, loss_norm_2, loss_1, loss_2, alpha, err_info, err_spread_type)

    # find the envelope constant function
    # if it is single mode, use the constant ceiling
    # if it is the rep code, use non constant

    row = block_size[1]
    col = block_size[2]

    if row == 1 && col == 1
        ceil_constant = 1.1
    else
       
        if block_no == 1
            ceil_constant = 1.5
        elseif block_no == 2
            ceil_constant = 1.5
        end
    end

    ceil_constant = 1.1
 
    counter = false

    # unpack loc
    x = loc[1]
    y = loc[2]
    z = loc[3]

    no_of_times = 0

    f_x = 0.0

    if block_no == 1
        while counter == false
            # sample a measurement
            samples[x, y, z] = sample_generator(code[1], measure_type[1], xbasis[1], N_ord, sum(loss_2[x, block_size[2]*j + y] for j = 0:block_size[3]-1), err_spread_type, 1)
            meas_exp[1][x, y, z], meas_exp[2][x, y, z], meas_exp[3][x, y, z], meas_exp[4][x, y, z] = meas_exp_prep(meas_ops[block_no], samples[x, y, z], err_prep_1[1][x, y, z], err_prep_1[2][x, y, z])

            f_x = pdf_1(meas_exp, err_exp_1, err_exp_2, norms, loc, block_size, loss_norm_1, loss_norm_2)

            if abs(f_x) == 0.0
                println("f_x is 0")
            end

            # sample a random number between 1 and max
            u = rand(Uniform(0, ceil_constant))

            # check if condition is true
            if abs(f_x) > ceil_constant
                println("goes above ", ceil_constant, ": ", abs(f_x), " | loc: ", loc)
                println(samples[x, y, z])
                sys.exit()
            else
                if u < abs(f_x)
                    no_of_times += 1
                    counter = true
                else 
                    no_of_times += 1
                end
            end
        end

    elseif block_no == 2
        while counter == false
            # sample a measurement
            samples[x, y, z] = sample_generator(code[2], measure_type[2], xbasis[2], N_ord, loss_1[x, mod(y, block_size[2])+1], err_spread_type, 2)
            meas_exp[1][x, y, z], meas_exp[2][x, y, z], meas_exp[3][x, y, z], meas_exp[4][x, y, z] = meas_exp_prep(meas_ops[block_no], samples[x, y, z], err_prep_2[1][x, y, z], err_prep_2[2][x, y, z])

            f_x = pdf_2(meas_exp_1, meas_exp, err_exp_2, norms, norms_1, loc, block_size, loss_norm_1, loss_norm_2)

            # sample a random number
            u = rand(Uniform(0, ceil_constant))

            # condition
            if abs(f_x) > ceil_constant
                println("goes above ", ceil_constant, ": ", abs(f_x), "| loc: ", loc)
                println(samples[x, y, z])
                println(meas_exp[1][x, y, z])
                sys.exit()
            else
                if u < abs(f_x)
                    no_of_times += 1
                    counter = true
                else
                    #println(abs(f_x))
                    no_of_times += 1
                end
            end
        end
    end

    norms[x, y, z] = f_x

    return samples[x, y, z], norms[x, y, z], meas_exp[1][x, y, z], meas_exp[2][x, y, z], meas_exp[3][x, y, z], meas_exp[4][x, y, z], 1/no_of_times
end

function sample_generator(code, meas_type, xbasis, N_ord, loss, err_spread_type, block)

    if meas_type == "heterodyne"

        if block == 1
            N_ord_main = N_ord[1]
            N_ord_sub = N_ord[2]
        else
            N_ord_main = N_ord[2]
            N_ord_sub = N_ord[1]
        end

        if code == "cat"
            edge = abs(xbasis[8])
        elseif code == "binomial"
            edge = sqrt(xbasis[8]*N_ord_main/2)
        end
        overflow = 1.75

        # sample over a circle for lesser surface area
        ### old system ###
        rad = rand(Uniform(0, edge + overflow))
        phi = rand(Uniform(0, 2*pi))
        beta = rad*(cos(phi) + 1im*sin(phi))

    elseif meas_type == "opt_phase" || meas_type == "adapt_homo"
        beta = rand(Uniform(0, 2*pi))
    end

    return beta
end

#######################
# Probability functions
function pdf_1(meas_exp_1, err_exp_1, err_exp_2, norms, loc, block_size, loss_norm_1, loss_norm_2)

    # setup all of the matrices
    no_row = block_size[1]
    no_col = block_size[2]
    rep = block_size[3]

    err_exp_1_plus = err_exp_1[1][:, :, loc[3]]
    err_exp_1_min = err_exp_1[2][:, :, loc[3]]
    err_exp_1_pm = err_exp_1[3][:, :, loc[3]]
    err_exp_1_mp = err_exp_1[4][:, :, loc[3]]

    err_exp_2_zero = err_exp_2[1][:, :, loc[3]]
    err_exp_2_one = err_exp_2[2][:, :, loc[3]]
    err_exp_2_zo = err_exp_2[3][:, :, loc[3]]
    err_exp_2_oz = err_exp_2[4][:, :, loc[3]]

    meas_exp_1_plus = meas_exp_1[1][:, :, loc[3]]
    meas_exp_1_min = meas_exp_1[2][:, :, loc[3]]
    meas_exp_1_pm = meas_exp_1[3][:, :, loc[3]]
    meas_exp_1_mp = meas_exp_1[4][:, :, loc[3]]

    norms = norms[:, :, loc[3]]
    x = loc[1] # corresponds to row number
    y = loc[2] # corresponds to col number
    z = loc[3] # corresponds to repetition number

    ### pdf begins ###
    ### B_factor ###
    B_plus = prod(prod(prod(err_exp_2_zero[i, (p-1)*no_col + j] for j = 1:no_col) + prod(err_exp_2_zo[i, (p-1)*no_col + j] for j = 1:no_col) + prod(err_exp_2_oz[i, (p-1)*no_col + j] for j = 1:no_col) + prod(err_exp_2_one[i, (p-1)*no_col + j] for j = 1:no_col) for p = 1:rep) for i = 1:no_row)/(2^(rep*no_row))
    B_min = prod(prod(prod(err_exp_2_zero[i, (p-1)*no_col + j] for j = 1:no_col) - prod(err_exp_2_zo[i, (p-1)*no_col + j] for j = 1:no_col) - prod(err_exp_2_oz[i, (p-1)*no_col + j] for j = 1:no_col) + prod(err_exp_2_one[i, (p-1)*no_col + j] for j = 1:no_col) for p = 1:rep) for i = 1:no_row)/(2^(rep*no_row))

    B_fac = B_plus + B_min
    
    ### A_factor ###
    A_plus = prod(prod(meas_exp_1_plus[i, j] for j = 1:no_col) + prod(meas_exp_1_pm[i, j] for j = 1:no_col) + prod(meas_exp_1_mp[i, j] for j = 1:no_col) + prod(meas_exp_1_min[i, j] for j = 1:no_col) for i = 1:x-1; init=1)*
            (prod(meas_exp_1_plus[x, j] for j = 1:y)*prod(err_exp_1_plus[x, j] for j = y+1:no_col; init=1) + prod(meas_exp_1_pm[x, j] for j = 1:y)*prod(err_exp_1_pm[x, j] for j = y+1:no_col; init=1) + prod(meas_exp_1_mp[x, j] for j = 1:y)*prod(err_exp_1_mp[x, j] for j = y+1:no_col; init=1) + prod(meas_exp_1_min[x, j] for j = 1:y)*prod(err_exp_1_min[x, j] for j = y+1:no_col; init=1))*
            prod(prod(err_exp_1_plus[i, j] for j = 1:no_col) + prod(err_exp_1_pm[i, j] for j = 1:no_col) + prod(err_exp_1_mp[i, j] for j = 1:no_col) + prod(err_exp_1_min[i, j] for j = 1:no_col) for i = x+1:no_row; init=1)/(2^no_row)

    A_min = prod(prod(meas_exp_1_plus[i, j] for j = 1:no_col) - prod(meas_exp_1_pm[i, j] for j = 1:no_col) - prod(meas_exp_1_mp[i, j] for j = 1:no_col) + prod(meas_exp_1_min[i, j] for j = 1:no_col) for i = 1:x-1; init=1)*
            (prod(meas_exp_1_plus[x, j] for j = 1:y)*prod(err_exp_1_plus[x, j] for j = y+1:no_col; init=1) - prod(meas_exp_1_pm[x, j] for j = 1:y)*prod(err_exp_1_pm[x, j] for j = y+1:no_col; init=1) - prod(meas_exp_1_mp[x, j] for j = 1:y)*prod(err_exp_1_mp[x, j] for j = y+1:no_col; init=1) + prod(meas_exp_1_min[x, j] for j = 1:y)*prod(err_exp_1_min[x, j] for j = y+1:no_col; init=1))*
            prod(prod(err_exp_1_plus[i, j] for j = 1:no_col) - prod(err_exp_1_pm[i, j] for j = 1:no_col) - prod(err_exp_1_mp[i, j] for j = 1:no_col) + prod(err_exp_1_min[i, j] for j = 1:no_col) for i = x+1:no_row; init=1)/(2^no_row)

    A_fac = A_plus + A_min

    ### norms ###
    current_norm = prod(norms)*prod(loss_norm_1)*prod(loss_norm_2)
    
    ### answer ###
    ans = A_fac*B_fac/(4*current_norm)

    return ans
end

function pdf_2(meas_exp_1, meas_exp_2, err_exp_2, norms, norms_1, loc, block_size, loss_norm_1, loss_norm_2)
    
    # setup all of the matrices
    no_row = block_size[1]
    no_col = block_size[2]
    rep = block_size[3]

    # location of current qubit
    x = loc[1] # row number
    y = loc[2] # col number
    z = loc[3]

    rho = Int(ceil(y/no_col)) # repetition number
    y_tru = Int(y - (rho - 1)*no_col) # row number in the repetition

    err_exp_2_zero = err_exp_2[1][:, :, loc[3]]
    err_exp_2_one = err_exp_2[2][:, :, loc[3]]
    err_exp_2_zo = err_exp_2[3][:, :, loc[3]]
    err_exp_2_oz = err_exp_2[4][:, :, loc[3]]

    # measurement expectation values
    meas_exp_1_plus = meas_exp_1[1][:, :, loc[3]]
    meas_exp_1_min = meas_exp_1[2][:, :, loc[3]]
    meas_exp_1_pm = meas_exp_1[3][:, :, loc[3]]
    meas_exp_1_mp = meas_exp_1[4][:, :, loc[3]]

    meas_exp_2_zero = meas_exp_2[1][:, :, loc[3]]
    meas_exp_2_one = meas_exp_2[2][:, :, loc[3]]
    meas_exp_2_zo = meas_exp_2[3][:, :, loc[3]]
    meas_exp_2_oz = meas_exp_2[4][:, :, loc[3]]

    norms = norms[:, :, loc[3]]
    norms_1 = norms_1[:, :, loc[3]]

    ### pdf begins ###
    ### A_factor ###
    A_plus = prod(prod(meas_exp_1_plus[i, j] for j = 1:no_col) + prod(meas_exp_1_pm[i, j] for j = 1:no_col) + prod(meas_exp_1_mp[i, j] for j = 1:no_col) + prod(meas_exp_1_min[i, j] for j = 1:no_col) for i = 1:no_row)/(2^no_row)
    A_min = prod(prod(meas_exp_1_plus[i, j] for j = 1:no_col) - prod(meas_exp_1_pm[i, j] for j = 1:no_col) - prod(meas_exp_1_mp[i, j] for j = 1:no_col) + prod(meas_exp_1_min[i, j] for j = 1:no_col) for i = 1:no_row)/(2^no_row)

    A_fac = A_plus + A_min

    ### B_factor ###
    B_plus = prod(prod(prod(meas_exp_2_zero[i, (p-1)*no_col + j] for j = 1:no_col) + prod(meas_exp_2_zo[i, (p-1)*no_col + j] for j = 1:no_col) + prod(meas_exp_2_oz[i, (p-1)*no_col + j] for j = 1:no_col) + prod(meas_exp_2_one[i, (p-1)*no_col + j] for j = 1:no_col) for p = 1:rep; init=1) for i = 1:x-1; init=1)*
                prod(prod(meas_exp_2_zero[x, (p-1)*no_col + j] for j = 1:no_col; init=1) + prod(meas_exp_2_zo[x, (p-1)*no_col + j] for j = 1:no_col; init=1) + prod(meas_exp_2_oz[x, (p-1)*no_col + j] for j = 1:no_col; init=1) + prod(meas_exp_2_one[x, (p-1)*no_col + j] for j = 1:no_col; init=1) for p = 1:rho-1; init=1)*
                (prod(meas_exp_2_zero[x, (rho-1)*no_col + j] for j = 1:y_tru; init=1)*prod(err_exp_2_zero[x, (rho-1)*no_col + j] for j = y_tru + 1:no_col; init=1) + prod(meas_exp_2_zo[x, (rho-1)*no_col + j] for j = 1:y_tru; init=1)*prod(err_exp_2_zo[x, (rho-1)*no_col + j] for j = y_tru + 1:no_col; init=1) + prod(meas_exp_2_oz[x, (rho-1)*no_col + j] for j = 1:y_tru; init=1)*prod(err_exp_2_oz[x, (rho-1)*no_col + j] for j = y_tru + 1:no_col; init=1) + prod(meas_exp_2_one[x, (rho-1)*no_col + j] for j = 1:y_tru; init=1)*prod(err_exp_2_one[x, (rho-1)*no_col + j] for j = y_tru+1:no_col; init=1))*
                prod(prod(err_exp_2_zero[x, (p-1)*no_col + j] for j = 1:no_col; init=1) + prod(err_exp_2_zo[x, (p-1)*no_col + j] for j = 1:no_col; init=1) + prod(err_exp_2_oz[x, (p-1)*no_col + j] for j = 1:no_col; init=1) + prod(err_exp_2_one[x, (p-1)*no_col + j] for j = 1:no_col; init=1) for p = rho + 1:rep; init=1)*
                prod(prod(prod(err_exp_2_zero[i, (p-1)*no_col + j] for j = 1:no_col) + prod(err_exp_2_zo[i, (p-1)*no_col+j] for j = 1:no_col) + prod(err_exp_2_oz[i, (p-1)*no_col+j] for j = 1:no_col) + prod(err_exp_2_one[i, (p-1)*no_col + j] for j = 1:no_col) for p = 1:rep; init=1) for i = x+1:no_row; init=1)/(2^(no_row*rep))

    B_min = prod(prod(prod(meas_exp_2_zero[i, (p-1)*no_col + j] for j = 1:no_col) - prod(meas_exp_2_zo[i, (p-1)*no_col + j] for j = 1:no_col) - prod(meas_exp_2_oz[i, (p-1)*no_col + j] for j = 1:no_col) + prod(meas_exp_2_one[i, (p-1)*no_col + j] for j = 1:no_col) for p = 1:rep; init=1) for i = 1:x-1; init=1)*
                prod(prod(meas_exp_2_zero[x, (p-1)*no_col + j] for j = 1:no_col; init=1) - prod(meas_exp_2_zo[x, (p-1)*no_col + j] for j = 1:no_col; init=1) - prod(meas_exp_2_oz[x, (p-1)*no_col + j] for j = 1:no_col; init=1) + prod(meas_exp_2_one[x, (p-1)*no_col + j] for j = 1:no_col; init=1) for p = 1:rho-1; init=1)*
                (prod(meas_exp_2_zero[x, (rho-1)*no_col + j] for j = 1:y_tru; init=1)*prod(err_exp_2_zero[x, (rho-1)*no_col + j] for j = y_tru + 1:no_col; init=1) - prod(meas_exp_2_zo[x, (rho-1)*no_col + j] for j = 1:y_tru; init=1)*prod(err_exp_2_zo[x, (rho-1)*no_col + j] for j = y_tru + 1:no_col; init=1) - prod(meas_exp_2_oz[x, (rho-1)*no_col + j] for j = 1:y_tru; init=1)*prod(err_exp_2_oz[x, (rho-1)*no_col + j] for j = y_tru + 1:no_col; init=1) + prod(meas_exp_2_one[x, (rho-1)*no_col + j] for j = 1:y_tru; init=1)*prod(err_exp_2_one[x, (rho-1)*no_col + j] for j = y_tru+1:no_col; init=1))*
                prod(prod(err_exp_2_zero[x, (p-1)*no_col + j] for j = 1:no_col; init=1) - prod(err_exp_2_zo[x, (p-1)*no_col + j] for j = 1:no_col; init=1) - prod(err_exp_2_oz[x, (p-1)*no_col + j] for j = 1:no_col; init=1) + prod(err_exp_2_one[x, (p-1)*no_col + j] for j = 1:no_col; init=1) for p = rho + 1:rep; init=1)*
                prod(prod(prod(err_exp_2_zero[i, (p-1)*no_col+j] for j = 1:no_col) - prod(err_exp_2_zo[i, (p-1)*no_col+j] for j = 1:no_col) - prod(err_exp_2_oz[i, (p-1)*no_col+j] for j = 1:no_col) + prod(err_exp_2_one[i, (p-1)*no_col + j] for j = 1:no_col) for p = 1:rep; init=1) for i = x+1:no_row; init=1)/(2^(no_row*rep))

    B_fac = B_plus + B_min

    ### norms ###
    current_norm = 1

    # get product of all block 1
    for i = 1:no_row
        for j = 1:no_col
            current_norm = current_norm*norms_1[i, j]*loss_norm_1[i, j]
        end
    end

    # get product of all block 2
    for i = 1:no_row
        for p = 1:rep
            for j = 1:no_col
                current_norm = current_norm*norms[i, (p-1)*no_col + j]*loss_norm_2[i, (p-1)*no_col + j]
            end
        end
    end

    ### answer ###
    ans = A_fac*B_fac/(4*current_norm)

    return ans
end