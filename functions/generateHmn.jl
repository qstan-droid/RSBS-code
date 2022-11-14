using JLD
using Combinatorics
using SpecialFunctions

###########################################
# functions that are useful
function gamma_spec(m, p)
    return sqrt(factorial(big(m)))/(2^p *factorial(big(m - 2*p))*factorial(big(p)))
end

function generalised_binom(alpha, n)
    return prod((alpha - k + 1)/k for k = 1:n;init=1)
end
###########################################
# what are the bounds
lmax = 100
lmax_bound = 0

# define dictionary
M = Dict((0, 0) => 1.0)

# define the M(n, 0) and M(0, n)
for n = 1:lmax + lmax_bound
    merge_dict_1 = Dict((n, 0) => 1/doublefactorial(2*n + 1))
    merge_dict_2 = Dict((0, n) => 1/doublefactorial(2*n + 1))
    merge!(M, merge_dict_1)
    merge!(M, merge_dict_2)
end

# define the other terms
for n = 1:lmax + lmax_bound
    for m = 1:lmax + lmax_bound
        term_1 = get!(M, (n-1, m), nothing)
        term_2 = get!(M, (n, m-1), nothing)

        # construct new term
        new_term = (n*term_1 + m*term_2)/(2*(n-m)^2 + n + m)
        merge!(M, Dict((n, m) => new_term))
    end
end

# plot dictionary??
M_array = zeros(Float64, (lmax+1, lmax+1))

for n = 0:lmax
    for m = 0:lmax
        M_array[n+1, m+1] = get!(M, (n, m), nothing)     
    end
end


println("finished with M_mn")

##############################
# Now we compute the Cpq terms
n_max = 20 # maximum Fock state support

# Have dictionary which has a matrix stored for every (n, m) term
C = Dict()

C_nm = zeros(Float64, (Int(ceil(n_max/2))+1, Int(ceil(n_max/2))+1))

# fill in the (p, q) for each n, m
for n = 0:n_max
    for m = 0:n_max

        for p = 0:Int(ceil(n_max/2))
            for q = 0:Int(ceil(n_max/2))
                C_nm[p+1, q+1] = sum(sum(generalised_binom((n-m)/2, l1)*generalised_binom((m-n)/2, l2)*get(M, (p+l1, q+l2), nothing) for l1 = 0:n_max) for l2 = 0:n_max)
            end
        end
        merge!(C, Dict((n, m) => C_nm))

    end
end

println("finished with C_pq")

# Now we construct the Hmn terms
H_mn = Dict()

for n = 0:n_max
    for m = 0:n_max
        C_matrix = get(C, (n, m), nothing)
        H_mn_val = sum(sum(gamma_spec(m, p)*gamma_spec(n, q)*C_matrix[p + 1, q + 1] for q = 0:Int(floor(n/2))) for p = 0:Int(floor(m/2))) 
        merge!(H_mn, Dict((m, n) => H_mn_val))
    end
end

# Put H_mn in array
H_mn_array = zeros(Float64, (n_max + 1, n_max + 1))

for n = 1:n_max+1
    for m = 1:n_max+1
        H_mn_array[n, m] = get(H_mn, (n-1, m-1), nothing)
    end
end

# then, save the array
save("H_mn.jld", "H_mn", H_mn_array)