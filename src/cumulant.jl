using IterTools
using QuadGK
using QuantumToolbox


struct overdampedBath
    T::Float64
    lam::Float64
    gamma::Float64
end

struct ohmicBath
    T::Float64
    alpha::Float64
    wc::Float64
end

function eigvects_to_kets(vects) #Maybe this is unnecessary
    return [Qobj(vects[:,i]) for i in 1:size(vects)[1]]
end

function decays(Hsys,Q,bath,t)
    ws,jump=jump_operators(Hsys,Q)
    combinations=collect(product(ws,ws))
    rates=Dict()
    for i in eachindex(combinations)
        j=combinations[i]
        rates[j]=decay.(j[1],j[2],bath,t) 
    end
    return rates
end

function cum_superop(Hsys,Q)
    ws,jump=jump_operators(Hsys,Q)
    combinations=collect(product(ws,ws))
    superops=Dict()
    for i in eachindex(combinations)
        j=combinations[i]
        superops[j]=spre(jump[j[2]])*spost(jump[j[1]]') - 0.5*(spre(jump[j[1]]' *jump[j[2]])+spost(jump[j[1]]' *jump[j[2]]))
    end
    return superops
end

function generator(Hsys,Q,bath,t)
    dict=decays(Hsys,Q,bath,t)#fine;
    dict2=cum_superop(Hsys,Q)#fine;
    ans=[0.0*Qobj(spre(Q)) for j in 1:length(t) ]
    for i in collect(keys(dict))
        ans +=[j *dict2[i] for j in dict[i]]
    end
    return ans
end

function csolve(Hsys,Q,bath,t,rho0)
    g=generator(Hsys,Q,bath,t)
    return [Qobj(vec2mat(exp(i).data * mat2vec(rho0.data))) for i in g]
end

function csolve(Hsys,Q::Array,bath::Array,t,rho0)
    gens=[]
    for (x,y) in zip(Q,bath)
        push!(gens,generator(Hsys,x,y,t))
    end
    g=sum(gens)
    return [Qobj(vec2mat(exp(i).data * mat2vec(rho0.data))) for i in g]
end

function jump_operators(Hsys,Q)
    eigvals,eigvects=eigen(Hsys);
    eigvects=eigvects_to_kets(eigvects);
    N=length(eigvals);
    collapse = []
    ws= []
    for j=1:N
        for k=j:N
            gap=eigvals[k] - eigvals[j]
            if  !isapprox(gap,0.0) #dephasing
                push!(ws,gap)
                push!(collapse,(eigvects[j]*eigvects[j]')*Q*(eigvects[k]*eigvects[k]'))
                push!(ws,-gap)
                push!(collapse,eigvects[k]*eigvects[k]'*Q*eigvects[j]*eigvects[j]')
            end
        end
    end
    push!(collapse,Q-sum(collapse))
    push!(ws,0)# dephasing
    for i in eachindex(collapse)
        if collapse[i]==Qobj(Hsys)*0
            deleteat!(ws,i)
            deleteat!(collapse,i)
        end
    end
    jump=Dict(ws .=> collapse)
    return ws,jump
end

function bose(Ω,T)
    return 1/(exp(Ω/T)-1)
end

function spectral_density(bath::overdampedBath,w)
    return 2*w*bath.lam*bath.gamma/(bath.gamma^2 + w^2)
end

function spectral_density(bath::ohmicBath,w)
    return bath.alpha*w*exp(abs(w)/bath.wc)
end

function power_spectrum(bath,w)
    return 2*(bose(w,bath.T)+1)*spectral_density(bath,w)
end
Base.Broadcast.broadcastable(q::overdampedBath) = Ref(q) 
Base.Broadcast.broadcastable(q::ohmicBath) = Ref(q) 

function gamma(ν,w,w1,bath,t)
    if ν==0
        return 0
    end
    var=t*t*exp(1im*(w-w1)/2 *t)*spectral_density(bath,ν)*(sinc((w-ν)/(2*pi) * t)*sinc((w1-ν)/(2*pi) * t))*(bose(ν,bath.T)+1)
    var+=t*t*exp(1im*(w-w1)/2 *t)*spectral_density(bath,ν)*(sinc((w+ν)/(2*pi) * t)*sinc((w1+ν)/(2*pi) * t))*bose(ν,bath.T)
    return var
end;

decay(w,w1,bath,t)=quadgk(ν -> gamma(ν,w,w1,bath,t), 0, Inf, rtol=1e-7,atol=1e-7)[1]

