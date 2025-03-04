#ifndef C_FUNC_FILE_H
#define C_FUNC_FILE_H

void compute_flux(double* rho_L, double* rho_R, double* vx_L, double* vx_R, double* vy_L, double* vy_R, double* P_L, 
                    double* P_R, double gamma, double* flux_Mass, double* flux_Momx, double* flux_Momy, double* flux_Energy, int dim);

#endif