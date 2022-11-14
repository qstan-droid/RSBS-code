using QuantumOptics
using Distributions

# Prepare cat codes of order N_ord with amplitude alpha
function code_prep(N_ord, dim, alpha, code)

    if code == "cat"
        b = FockBasis(dim)

        zero_cat = normalize!(sum(coherentstate(b, alpha * exp((1im * i * pi) / N_ord)) for i = 0:2*N_ord-1))
        one_cat = normalize!(sum(coherentstate(b, alpha * exp((1im * i * pi) / N_ord)) * (-1)^i for i = 0:2*N_ord-1))

        plus_cat = (zero_cat + one_cat)/sqrt(2)
        min_cat = (zero_cat - one_cat)/sqrt(2)

        n_b = number(b)
        a_b = destroy(b)

        prep_state = [plus_cat, min_cat, n_b, a_b, zero_cat, one_cat, b, alpha, dim]

    elseif code == "binomial"
        b = FockBasis(dim)

        plus_bin = (sum(sqrt((1/(2^(alpha)))*(binomial(alpha, k)))*fockstate(b, k*N_ord) for k = 0:alpha))
        min_bin = (sum((-1)^k * sqrt((1/(2^(alpha)))*(binomial(alpha, k)))*fockstate(b, k*N_ord) for k = 0:alpha))

        zero_bin = (plus_bin + min_bin)/(sqrt(2))
        one_bin = (plus_bin - min_bin)/(sqrt(2))

        n_b = number(b)
        a_b = destroy(b)

        prep_state = [plus_bin, min_bin, n_b, a_b, zero_bin, one_bin, b, alpha, dim]
    end

    return prep_state
end

# find the dimensions based on alpha
function find_dim(code, alpha, N_ord)

    if code == "cat"
        if alpha <= 1
            dim = 10
        else
            dim = convert(Int64, round(2*alpha^2 + alpha, digits=0))
        end
    elseif code == "binomial"
        dim = (alpha)*(N_ord)
    end

    return dim
end
