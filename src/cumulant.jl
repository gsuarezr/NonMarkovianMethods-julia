
using QuadGK

struct cumulant
    Hsys
    t::Float64
    eps::Float64
    baths
    Qs
end

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

decay(w,w1,bath,t)=quadgk(ν -> gamma(ν,w,w1,bath,t), 0, Inf, rtol=1e-6,atol=1e-6)[1]
