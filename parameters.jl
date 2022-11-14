#------------------------
# functions
logrange(x1, x2, n) = (10^y for y in range(log10(x1), log10(x2), length=n))

#-------------------------
# number of samples
sample_no = 1

# type of measurement
# heterodyne, opt_phase, adapt_homo
measure = ["opt_phase", "opt_phase"]

# note the invariable parameters of code
code = ["binomial", "binomial"]
block_size = [1, 3, 3] # row, col, rep
N_ord = [2, 2]

# define what errors are, how strong and where they're applied
# error_placement = [loss on block_1, dephase on block_1, loss on block_2, dephase on block_2]
# error_info = [nu_loss_1, nu_dephase_1, nu_loss_2, nu_dephase_2]
err_place = [true, false, true, false]
err_info = [0.01, 0.0, 0.01, 0.0]

# choose to vary alpha or bias
x_var = "alpha"

if x_var == "alpha"
    bias = [0, 0]

    x_min = 2
    x_step = 1
    x_max = 6

    x = x_min:x_step:x_max
    # same alpha for both blocks
    dif_alpha = false
    alpha_2 = 15
elseif x_var == "bias"
    alpha = [5, 15]
    where_bias = 2

    x_min = 0
    x_step = 0.1
    x_max = 2*pi

    x = x_min:x_step:x_max
elseif x_var == "gamma"
    alpha = [4, 4]

    x_min = -3
    x_num = 15
    x_max = 0

    x = 10 .^ range(x_min, stop = x_max, length = x_num)
end

# how to decode
# decoder_type = naive, naive_ave_bias, ml, ml_ave, mlnc
decode_type = "ml_ave"
err_spread_type = "normal" # normal, no_spread, err_trans
