#include <stdio.h>
#include <math.h>
#include "flux.h"

double max(double first, double second) {
    if (first < second) {
        return second;
    } else {
        return first;
    }
}

double abs_val(double value) {
    if (value < 0) {
        return -1.0 * value;
    } 
    return value;
}

/**
 * Iterate over 2d arrays even though we just have pointer. This is because we ensure that the numpy arrays 
 * we pass here are in C ORDER (for more, read here: https://stackoverflow.com/questions/26998223/what-is-the-difference-between-contiguous-and-non-contiguous-arrays)
 * 
 * Everything is stored in contiguous block of memory, so we go over the rows first, AND THEN we go over the columns
 */
void compute_flux(double* rho_L, double* rho_R, double* vx_L, double* vx_R, double* vy_L, double* vy_R, double* P_L, 
                    double* P_R, double gamma, double* flux_Mass, double* flux_Momx, double* flux_Momy, double* flux_Energy, int dim) {
    
    for (int row = 0; row < dim; row++) {
        for(int col = 0; col < dim; col++) {
            int i = (row * dim) + col;

            double en_L = P_L[i]/(gamma-1)+0.5*rho_L[i] * (vx_L[i]*vx_L[i] + vy_L[i]*vy_L[i]);
            double en_R = P_R[i]/(gamma-1)+0.5*rho_R[i] * (vx_R[i]*vx_R[i] + vy_R[i]*vy_R[i]);

            double rho_star = 0.5*(rho_L[i] + rho_R[i]);
            double momx_star = 0.5*(rho_L[i] * vx_L[i] + rho_R[i] * vx_R[i]);
            double momy_star = 0.5*(rho_L[i] * vy_L[i] + rho_R[i] * vy_R[i]);
            double en_star = 0.5*(en_L + en_R);

            double P_star =  (gamma-1)*(en_star - 0.5*(momx_star*momx_star+momy_star*momy_star)/rho_star);

            double C = max(
                sqrt(gamma*P_L[i]/rho_L[i]) + abs_val(vx_L[i]), 
                sqrt(gamma*P_R[i]/rho_R[i]) + abs_val(vx_R[i])
            );

            flux_Mass[i] = momx_star - (C * 0.5 * (rho_L[i] - rho_R[i]));
            flux_Momx[i] = (momx_star*momx_star/rho_star + P_star) - (C * 0.5 * (rho_L[i] * vx_L[i] - rho_R[i] * vx_R[i]));
            flux_Momy[i] = (momx_star * momy_star/rho_star) - (C * 0.5 * (rho_L[i] * vy_L[i] - rho_R[i] * vy_R[i]));
            flux_Energy[i] = ((en_star+P_star) * momx_star/rho_star) - (C * 0.5 * (en_L - en_R));            
        }
    }

}