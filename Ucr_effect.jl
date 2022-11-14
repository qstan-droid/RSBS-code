using QuantumOptics
using Plots
using NIntegration
using DelimitedFiles
using Plots
import PyPlot

const plt = PyPlot

include("functions/code_prep.jl")
include("functions/measurement.jl")
include("err_trans_single_mode.jl")

#------------------------------------#
# codeword variables
code = ["binomial", "binomial"]

N_ord = [2, 2]
alpha = [2, 2]

nu = [0.01, 0.01]

# create codewords
dim_1 = find_dim(code[1], alpha[1], N_ord[1])
dim_2 = find_dim(code[2], alpha[2], N_ord[2])

xbasis_1 = code_prep(N_ord[1], dim_1, alpha[1], code[1])
xbasis_2 = code_prep(N_ord[2], dim_2, alpha[2], code[2])

b1 = xbasis_1[7]
b2 = xbasis_2[7]

# prepare the U_cr

dn = min(N_ord[1], N_ord[2])
gt = pi # radians of rotation

# upper bounds for U_cr
up_bound_1 = Int(floor(alpha[1]/2) - 1)
up_bound_2 = Int(floor(alpha[2]/2) - 1)

if up_bound_1 <= 0
    up_bound_1 = 0
end

if up_bound_2 <= 0
    up_bound_2 = 0
end

U_cr = tensor(identityoperator(b1), identityoperator(b2)) + (exp(1im*gt) - 1)*sum(sum(sum(sum(tensor(projector(fockstate(b1, (2*j+1)*N_ord[1] - la)), projector(fockstate(b2, (2*k+1)*N_ord[2] - lb))) for lb = 0:dn-1-la) for la = 0:dn-1) for j = 0:up_bound_1) for k = 0:up_bound_2)
U_trans = exp(dense((1im*pi/(N_ord[1]*N_ord[2]))*tensor((xbasis_1[3]), (xbasis_2[3]))))

# the proper one?
g_cr = pi/2
t = 2
#H_cr = -g_cr*sum(sum(tensor(projector(fockstate(b1, (2*j+1)*N_ord[1] - la)), sum(projector(fockstate(b2, (2*j+1)*N_ord[2] - lb)) for lb = 0:dn-1-la)) for la = 0:dn-1) for j = 0:up_bound_1) 

#U_cr_var = exp(-1im*H_cr*t)

#------------------------------------#
# loss error 
E = function(x::Int64, nu, n_b, a_b)
    (((1 - exp(-nu))^(x/2))/(sqrt(factorial(big(x)))))*exp(-nu/2*dense(n_b))*(a_b^x)
end

# figure out the size of the commutator by changing it into a 4 dimensional vector
zero_1 = xbasis_1[5] 
one_1 = xbasis_1[6]

zero_2 = xbasis_2[5]
one_2 = xbasis_2[6]

plus_1 = xbasis_1[1]
min_1 = xbasis_1[2]

plus_2 = xbasis_2[1]
min_2 = xbasis_2[2]

states = [tensor(zero_1, zero_2), tensor(zero_1, one_2), tensor(one_1, zero_2), tensor(one_1, one_2)]
states_2 = [tensor(plus_1, plus_2), tensor(plus_1, min_2), tensor(min_1, plus_2), tensor(min_1, min_2)]

#loss = 0:5
#norm_new = zeros(Float64, length(loss))
#norm_old = zeros(Float64, length(loss))


# create the commutator

#zeta = U_cr*tensor(E(l, nu[1], xbasis_1[3], xbasis_1[4]), identityoperator(b2))*dagger(U_cr)

# old U_trans
#matrix = zeros(ComplexF64, 4, 4)
#for i = 1:4
#    for j = 1:4
#        matrix[i, j] = dagger(states_2[j])*(zeta - tensor(E(l, nu[1], xbasis_1[3], xbasis_1[4]), identityoperator(b2)))*states_2[i]
#    end
#end
#norm_new = norm(matrix)

#println(zeta*tensor(plus_1, plus_2))
println("--------------------------------------")
# look at how norm changes with 

# finding husini-q functions
x = -5:0.1:5
y = -5:0.1:5

l = 4
nu_pow = -2
nu = 10.0^(nu_pow)

plot_data = zeros(Float64, length(x), length(y))

zeta = U_cr*tensor(E(l, nu, xbasis_1[3], xbasis_1[4]), identityoperator(b2))*dagger(U_cr)
zeta_old = U_trans*tensor(E(l, nu, xbasis_1[3], xbasis_1[4]), identityoperator(b2))*dagger(U_trans)

lop = normalize(ptrace(zeta*tensor(plus_2, plus_2), 2))

for i = 1:length(x)
    for j = 1:length(y)
        plot_data[i, j] = abs(dagger(coherentstate(b2, x[i]+1im*y[j]))*lop*coherentstate(b2, x[i]+1im*y[j]))/pi
    end
end

img = heatmap(x, y, plot_data)
#savefig(img, string("imaging_pdf/CPHASE_mode=1_F_l=", l, "_nu=", nu_pow, "_N=2"))
#--------------------------------------------------#
# functions for the first and second mode

#l = 0
#nu = 0.01

#lop = normalize(ptrace((U_cr*tensor(E(l, nu, xbasis_1[3], xbasis_1[4]), identityoperator(b2))*dagger(U_cr) - tensor(E(l, nu, xbasis_1[3], xbasis_1[4]), identityoperator(b2)))*tensor(plus_1, plus_2), 2))

#mode1 = function(r, phi)
#    dagger(coherentstate(b1, r*exp(1im*phi))*lop*coherentstate(b1, r*exp(1im*phi)))/pi
#end

#mode2 = function(r, phi)
#    dagger(coherentstate(b2, r*exp(1im*phi))*lop*coherentstate(b2, r*exp(1im*phi)))/pi
#end

# probability we get it right
#r_max = 5

#p1 = nintegrate(mode1, (0, 0), (r_max, pi/4)) + nintegrate(mode1, (0, 0), (r_max, -pi/4))

#println(p1)

## prepare measurement
#meas_1 = measurement_operator("heterodyne", xbasis_1, N_ord[1], 0)
#meas_2 = measurement_operator("heterodyne", xbasis_2, N_ord[2], 0)

## get function of the thing

#p = function(x1, x2, l::Int64)
#    abs(tr(tensor(meas_1(x1), meas_2(x2))*zeta*tensor(dm(plus_1), dm(plus_2))*dagger(zeta))/(dagger(zeta*tensor(plus_1, plus_2))*zeta*tensor(plus_1, plus_2)))
#end

#------------------------------------------#
#N_ord = 2
#
#K = 1:5
#n_ave = N_ord .* K ./ 2
#
#println(K)
## import data
#gate_infid = []
#gate_err = []
#
#infid_import = 1 .- readdlm(string(@__DIR__, "\\trans_single_mode_data\\data_gate.csv"), ',', Float64)
#infid_err = readdlm(string(@__DIR__, "\\trans_single_mode_data\\data_gate_err.csv"), ',', Float64)
#push!(gate_infid, infid_import)
#push!(gate_err, infid_err)
#infid_import = 1 .- readdlm(string(@__DIR__, "\\data_naive_0.01_same_err_no_diff_N2\\gate_het.csv"), ',', Float64)
#infid_err = readdlm(string(@__DIR__, "\\data_naive_0.01_same_err_no_diff_N2\\gate_SE_het.csv"), ',', Float64)
#push!(gate_infid, infid_import)
#push!(gate_err, infid_err)
#infid_import = 1 .- readdlm(string(@__DIR__, "\\data_naive_0.01_same_err_no_diff_N2_no_spread\\gate_het.csv"), ',', Float64)
#infid_err = readdlm(string(@__DIR__, "\\data_naive_0.01_same_err_no_diff_N2\\gate_SE_het.csv"), ',', Float64)
#push!(gate_infid, infid_import)
#push!(gate_err, infid_err)
#
#no_of_samples = 1000
#
#println(gate_infid)
#
##### Break Even #####
#nu = 0.01
#break_even = 1 - ((1 + exp(-nu/2))^2 /2 + 1)/3
#
#p = plot(n_ave,
#        [gate_infid[1][K] gate_infid[2][K] gate_infid[3][K]],
#        yerror = [gate_err[1][K] gate_err[2][K] gate_err[3][K]],
#        color = [:red :blue :green :grey],
#        xlabel = "\$\\hat{n}\$",
#        ylabel = "Average Infidelity",
#        label = ["error transparent" "CPHASE" "ideal error transparent CPHASE"])
#Plots.abline!(0, break_even, line=:dash, color=:grey, label = "break even")
#
#plot(p)
#savefig(p, string(@__DIR__, "\\imaging_pdf\\comparing_gates_het.png"))