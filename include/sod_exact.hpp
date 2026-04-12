#pragma once
// Exact solution to the Sod shock tube Riemann problem.
// Ref: Toro (2009) §4.3 — iterative p* solver via Newton-Raphson.
//
// Initial condition (non-dim):
//   x < 0.5:  rho=1.0, u=0, p=1.0   (left)
//   x >= 0.5: rho=0.125, u=0, p=0.1 (right)
// gamma = 1.4

#include <cmath>
#include <array>

struct SodSample {
    double rho, u, p;
};

inline SodSample sod_exact_sample(double x, double t,
                                   double x0 = 0.5,
                                   double rhoL=1.0,   double uL=0.0, double pL=1.0,
                                   double rhoR=0.125, double uR=0.0, double pR=0.1,
                                   double gamma=1.4)
{
    const double aL = std::sqrt(gamma*pL/rhoL);
    const double aR = std::sqrt(gamma*pR/rhoR);

    // Newton-Raphson for p*
    auto f = [&](double p, double rho_k, double p_k, double a_k) -> double {
        if (p > p_k) {
            double A = 2.0/((gamma+1.0)*rho_k);
            double B = (gamma-1.0)/(gamma+1.0)*p_k;
            return (p-p_k)*std::sqrt(A/(p+B));
        } else {
            return 2.0*a_k/(gamma-1.0)*(std::pow(p/p_k,(gamma-1.0)/(2.0*gamma))-1.0);
        }
    };
    auto df = [&](double p, double rho_k, double p_k, double a_k) -> double {
        if (p > p_k) {
            double A = 2.0/((gamma+1.0)*rho_k);
            double B = (gamma-1.0)/(gamma+1.0)*p_k;
            double Q = std::sqrt(A/(p+B));
            return Q - (p-p_k)*0.5*A/((p+B)*(p+B))/Q*(p+B);
        } else {
            return 1.0/(rho_k*a_k)*std::pow(p/p_k,-(gamma+1.0)/(2.0*gamma));
        }
    };

    double p_star = 0.5*(pL+pR);
    for (int iter=0; iter<100; ++iter) {
        double F  = f(p_star,rhoL,pL,aL) + f(p_star,rhoR,pR,aR) + (uR-uL);
        double Fp = df(p_star,rhoL,pL,aL) + df(p_star,rhoR,pR,aR);
        double dp = -F/Fp;
        p_star += dp;
        if (p_star < 1e-10) p_star = 1e-10;
        if (std::abs(dp) < 1e-10*(p_star+1e-300)) break;
    }
    double u_star = 0.5*(uL+uR) + 0.5*(f(p_star,rhoR,pR,aR) - f(p_star,rhoL,pL,aL));

    // Density in star regions
    double rhoL_star, rhoR_star;
    if (p_star > pL) {
        // Left: shock
        rhoL_star = rhoL*(p_star/pL + (gamma-1.0)/(gamma+1.0)) /
                         ((gamma-1.0)/(gamma+1.0)*p_star/pL + 1.0);
    } else {
        rhoL_star = rhoL*std::pow(p_star/pL, 1.0/gamma);
    }
    if (p_star > pR) {
        // Right: shock
        rhoR_star = rhoR*(p_star/pR + (gamma-1.0)/(gamma+1.0)) /
                         ((gamma-1.0)/(gamma+1.0)*p_star/pR + 1.0);
    } else {
        rhoR_star = rhoR*std::pow(p_star/pR, 1.0/gamma);
    }

    // Wave speeds
    double SL=0, SR=0, SHL=0, STL=0, SHR=0, STR=0;
    if (p_star > pL) {
        // Left shock
        SL = uL - aL*std::sqrt((gamma+1.0)/(2.0*gamma)*p_star/pL + (gamma-1.0)/(2.0*gamma));
    } else {
        // Left rarefaction
        double aL_star = aL*std::pow(p_star/pL,(gamma-1.0)/(2.0*gamma));
        SHL = uL - aL;
        STL = u_star - aL_star;
    }
    if (p_star > pR) {
        // Right shock
        SR = uR + aR*std::sqrt((gamma+1.0)/(2.0*gamma)*p_star/pR + (gamma-1.0)/(2.0*gamma));
    } else {
        // Right rarefaction
        double aR_star = aR*std::pow(p_star/pR,(gamma-1.0)/(2.0*gamma));
        SHR = uR + aR;
        STR = u_star + aR_star;
    }

    if (t < 1e-300) return {(x < x0) ? rhoL : rhoR,
                            (x < x0) ? uL   : uR,
                            (x < x0) ? pL   : pR};

    double xi = (x - x0) / t;   // self-similar variable

    // Sample the solution
    if (p_star > pL) {
        // Left: shock wave
        if (xi < SL)       return {rhoL,      uL,     pL};
        else if (xi < u_star) return {rhoL_star, u_star, p_star};
    } else {
        // Left: rarefaction fan
        if (xi < SHL)       return {rhoL, uL, pL};
        else if (xi < STL) {
            double aL_star = aL*std::pow(p_star/pL,(gamma-1.0)/(2.0*gamma));
            double u_fan = 2.0/(gamma+1.0)*(aL + (gamma-1.0)/2.0*uL + xi);
            double a_fan = u_fan - xi + aL - (gamma-1.0)/2.0*uL - aL; // simplify
            a_fan = aL - (gamma-1.0)/2.0*(u_fan - uL);
            double rho_fan = rhoL*std::pow(a_fan/aL, 2.0/(gamma-1.0));
            double p_fan   = pL*std::pow(a_fan/aL, 2.0*gamma/(gamma-1.0));
            (void)aL_star;
            return {rho_fan, u_fan, p_fan};
        } else if (xi < u_star) return {rhoL_star, u_star, p_star};
    }

    if (xi < u_star) return {rhoL_star, u_star, p_star};  // fallthrough contact

    if (p_star > pR) {
        // Right: shock wave
        if (xi < SR)   return {rhoR_star, u_star, p_star};
        else           return {rhoR, uR, pR};
    } else {
        // Right: rarefaction fan
        if (xi < STR)  return {rhoR_star, u_star, p_star};
        else if (xi < SHR) {
            double a_fan = aR + (gamma-1.0)/2.0*(uR - xi);
            double u_fan = 2.0/((gamma+1.0))*(-aR + (gamma-1.0)/2.0*uR + xi);
            double rho_fan = rhoR*std::pow(a_fan/aR, 2.0/(gamma-1.0));
            double p_fan   = pR*std::pow(a_fan/aR, 2.0*gamma/(gamma-1.0));
            return {rho_fan, u_fan, p_fan};
        } else         return {rhoR, uR, pR};
    }
}
