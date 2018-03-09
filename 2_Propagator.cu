# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <cuda_runtime.h>
# include <device_launch_parameters.h>
# include "time.h"
# define PI 3.141592653589793
# define TILE_WIDTH_X 512
# define I(ix,iz) (ix)+nx*(iz)


__global__ void CPML_x(float *a_x, float *b_x, int CPML, float Vmax, int nx, float dt, float dh, float f, float d0, float L)
{

	int ix = threadIdx.x + blockDim.x * blockIdx.x;

	if (ix < CPML)
	{
	    b_x[ix] = exp(-(d0*Vmax*pow(((CPML-ix)*dh/L),2) + PI*f*(L-(CPML-ix)*dh)/L)*dt);
	    a_x[ix] = d0*Vmax*pow((CPML-ix)*dh/L,2)*(b_x[ix]-1)/((d0*Vmax*pow((CPML-ix)*dh/L,2) + PI*f*(L-(CPML-ix)*dh)/L));
	}
	
	if (ix > (nx-CPML-1) && ix < nx)
	{
	    b_x[ix] = exp(-(d0*Vmax*pow((ix-nx+CPML+1)*dh/L,2) + PI*f*(L-(ix-nx+CPML+1)*dh)/L)*dt);
	    a_x[ix] = d0*Vmax*pow((ix-nx+CPML+1)*dh/L,2)*(b_x[ix]-1)/((d0*Vmax*pow((ix-nx+CPML+1)*dh/L,2) + PI*f*(L-(ix-nx+CPML+1)*dh)/L));
	}
}

__global__ void CPML_z(float *a_z, float *b_z, int CPML, float Vmax, int nz, float dt, float dh, float f, float d0, float L)
{

	int iz = threadIdx.x + blockDim.x * blockIdx.x;

	if (iz > (nz-CPML-1) && iz < nz)
	{
	    b_z[iz] = exp(-(d0*Vmax*pow((iz-nz+CPML+1)*dh/L,2) + PI*f*(L-(iz-nz+CPML+1)*dh)/L)*dt);
	    a_z[iz] = d0*Vmax*pow((iz-nz+CPML+1)*dh/L,2)*(b_z[iz]-1)/((d0*Vmax*pow((iz-nz+CPML+1)*dh/L,2) + PI*f*(L-(iz-nz+CPML+1)*dh)/L));
	}
}

__global__ void PSI_F (float *P, float *a_x, float *b_x, float *a_z, float *b_z, float *psi_px, float *psi_pz, int nx, int nz, float dh, int CPML)
{

	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iz = threadIdx.y + blockDim.y * blockIdx.y;
	
	float temp_px = 0, temp_pz = 0;
	if (ix > 0 && ix < (nx-1) && iz > 0 && iz < (nz-1) && (ix < CPML || ix > (nx-CPML-1) || iz > (nz-CPML-1)))
	{
	    temp_px = (P[I(ix+1,iz)] - P[I(ix-1,iz)])/(2*dh);
	    temp_pz = (P[I(ix,iz+1)] - P[I(ix,iz-1)])/(2*dh);
	    
	    psi_px[I(ix,iz)] = b_x[ix]*psi_px[I(ix,iz)] + a_x[ix]*temp_px;
	    psi_pz[I(ix,iz)] = b_z[iz]*psi_pz[I(ix,iz)] + a_z[iz]*temp_pz;
	}
}

__global__ void ZETA_F (float *P, float *a_x, float *b_x, float *a_z, float *b_z, float *psi_px, float *psi_pz, float *z_px, float *z_pz, float *aten_px, float *aten_pz, int nx, int nz, float dh, float dh2, int CPML)
{

	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iz = threadIdx.y + blockDim.y * blockIdx.y;
	
	float temp_px = 0, temp_pz = 0, temp_x = 0, temp_z = 0;
	if (ix > 0 && ix < (nx-1) && iz > 0 && iz < (nz-1) && (ix < CPML || ix > (nx-CPML-1) || iz > (nz-CPML-1)))
	{
	    temp_x = (psi_px[I(ix+1,iz)] - psi_px[I(ix-1,iz)])/(2*dh);
	    temp_z = (psi_pz[I(ix,iz+1)] - psi_pz[I(ix,iz-1)])/(2*dh);
	    
	    temp_px = (P[I(ix+1,iz)] - 2.0*P[I(ix,iz)] + P[I(ix-1,iz)])/dh2 + temp_x;
	    temp_pz = (P[I(ix,iz+1)] - 2.0*P[I(ix,iz)] + P[I(ix,iz-1)])/dh2 + temp_z;
	    
	    z_px[I(ix,iz)] = b_x[ix]*z_px[I(ix,iz)] + a_x[ix]*temp_px;
	    z_pz[I(ix,iz)] = b_z[iz]*z_pz[I(ix,iz)] + a_z[iz]*temp_pz;
	    
	    aten_px[I(ix,iz)] = temp_x + z_px[I(ix,iz)];
	    aten_pz[I(ix,iz)] = temp_z + z_pz[I(ix,iz)];
	}
}

__global__ void propagator_F (float *P_pre, float *P_past, float *vel, float *aten_px, float *aten_pz, float *source, float *shot, float C, float dh2, int nx, int nz, int nt, int sx, int sz, int it, int is, int CPML, int flag)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iz = threadIdx.y + blockDim.y * blockIdx.y;

	float temp_p = 0;
	if (ix > 0 && ix < (nx-1) && iz > 0 && iz < (nz-1))
	{
	    temp_p = P_pre[I(ix+1,iz)] - 4.0*P_pre[I(ix,iz)] + P_pre[I(ix-1,iz)] + P_pre[I(ix,iz+1)] + P_pre[I(ix,iz-1)] + dh2*aten_px[I(ix,iz)] + dh2*aten_pz[I(ix,iz)];
	    P_past[I(ix,iz)] = 2*P_pre[I(ix,iz)] - P_past[I(ix,iz)] + pow(vel[I(ix,iz)],2)*C*temp_p;	    
	}

	if (ix == (sx-1) && iz == (sz-1)) 
	{
	    P_past[I(ix,iz)] += source[it];
	}
	
	if (ix > (CPML-1) && ix < (nx-CPML) && iz == (sz-1))
	{
	    shot[I(ix,it)] = P_past[I(ix,iz)];
	}
}


int main ()
{
  
    /******************* PARAMETERS *********************/
    int CPML = 20;
    int nx = 301;
    int nz = 120;
    float dh = 20;
    float dt = 2e-3;
    float tend = 3;
    int nt = ceil(tend/dt);
    int sx = 150, sz = 3;
    
    float R = 1e-6;
    float L = CPML*dh;
    float d0 = -3*log(R)/(2*L);
    float Vmax = 3000;
    float f = 15;

    float dh2 = dh*dh;
    float C = (dt*dt)/dh2;
    
    float *source_h = (float*)calloc(nt,sizeof(float));
    float *O_vel_h = (float*)calloc(nx*nz,sizeof(float));
    float *shot_h = (float*)calloc(nx*nt, sizeof(float));
    float *P = (float*)calloc(nx*nz*nt, sizeof(float));
    
    
    /*************** LOAD SOURCE AND MODEL ****************/
    int it = 0;
    float t = 0;
    for (it=0;it<nt;it++)
    {
	source_h[it] = (1.0-2.0*pow(PI*f*(t-(1.0/f)),2))*exp(-pow(PI*f*(t-(1.0/f)),2));
	t += dt;
    }
    
    FILE *M = fopen("v_ori.bin","rb");
    fread(O_vel_h,nx*nz,sizeof(float),M);
    fclose(M);
    
    
    clock_t begin, end;
    float time_spent;
    
    /********************** CUDA ***************************/
    float *source, *vel_O, *P_pre, *P_past, *shot;
    float *a_x, *a_z, *b_x, *b_z, *psi_px, *psi_pz, *z_px, *z_pz, *aten_px, *aten_pz;
    
    cudaMalloc((void **) &source, nt*sizeof(float));
    cudaMalloc((void **) &vel_O, nx*nz*sizeof(float));
    
    cudaMalloc((void **) &a_x, nx*sizeof(float));
    cudaMalloc((void **) &a_z, nz*sizeof(float));
    cudaMalloc((void **) &b_x, nx*sizeof(float));
    cudaMalloc((void **) &b_z, nz*sizeof(float));
    cudaMalloc((void **) &psi_px, nx*nz*sizeof(float));
    cudaMalloc((void **) &psi_pz, nx*nz*sizeof(float));
    cudaMalloc((void **) &z_px, nx*nz*sizeof(float));
    cudaMalloc((void **) &z_pz, nx*nz*sizeof(float));
    cudaMalloc((void **) &aten_px, nx*nz*sizeof(float));
    cudaMalloc((void **) &aten_pz, nx*nz*sizeof(float));

    cudaMalloc((void **) &P_pre, nx*nz*sizeof(float));
    cudaMalloc((void **) &P_past, nx*nz*sizeof(float));
    cudaMalloc((void **) &shot, nx*nt*sizeof(float));
    
    cudaMemcpy(vel_O, O_vel_h, nx*nz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(source, source_h, nt*sizeof(float), cudaMemcpyHostToDevice);
        
    cudaMemset(a_x, 0, nx*sizeof(float));
    cudaMemset(b_x, 0, nx*sizeof(float));
    cudaMemset(a_z, 0, nz*sizeof(float));
    cudaMemset(b_z, 0, nz*sizeof(float));
    cudaMemset(psi_px, 0, nx*nz*sizeof(float));
    cudaMemset(psi_pz, 0, nx*nz*sizeof(float));
    cudaMemset(z_px, 0, nx*nz*sizeof(float));
    cudaMemset(z_pz, 0, nx*nz*sizeof(float));
    cudaMemset(aten_px, 0, nx*nz*sizeof(float));
    cudaMemset(aten_pz, 0, nx*nz*sizeof(float));
    cudaMemset(P_pre, 0, nx*nz*sizeof(float));
    cudaMemset(P_past, 0, nx*nz*sizeof(float));
    cudaMemset(shot, 0, nx*nt*sizeof(float));
    
    dim3 Grid_CPML_x(ceil(nx/(float)TILE_WIDTH_X));
    dim3 Grid_CPML_z(ceil(nz/(float)TILE_WIDTH_X));
    dim3 Block_CPML(TILE_WIDTH_X);
    
    dim3 Grid(ceil(nx/(float)32),ceil(nz/(float)32));
    dim3 Block(32,32);
    
    begin = clock();
    /******************* KERNELS *******************/

    CPML_x <<<Grid_CPML_x,Block_CPML>>>(a_x, b_x, CPML, Vmax, nx, dt, dh, f, d0, L);
    CPML_z <<<Grid_CPML_z,Block_CPML>>>(a_z, b_z, CPML, Vmax, nz, dt, dh, f, d0, L);
    
    for (it=0;it<nt;it++)
    {
	PSI_F <<<Grid,Block>>>(P_pre, a_x, b_x, a_z, b_z, psi_px, psi_pz, nx, nz, dh, CPML);
	ZETA_F <<<Grid,Block>>>(P_pre, a_x, b_x, a_z, b_z, psi_px, psi_pz, z_px, z_pz, aten_px, aten_pz, nx, nz, dh, dh2, CPML);
	propagator_F <<<Grid,Block>>>(P_pre, P_past, vel_O, aten_px, aten_pz, source, shot, C, dh2, nx, nz, nt, sx, sz, it, 0, CPML, 0);
	cudaMemcpy(P+(it*nx*nz), P_past, nx*nz*sizeof(float), cudaMemcpyDeviceToHost);

	float *URSS = P_past;
	P_past = P_pre;
	P_pre = URSS;
    }

    
    end = clock();
    time_spent = (float)(end - begin) / CLOCKS_PER_SEC;
    printf("time: %e\n", time_spent);
    
    cudaMemcpy(shot_h, shot, nx*nt*sizeof(float),cudaMemcpyDeviceToHost);
    
    FILE *shotgather;
    shotgather=fopen("tobs.bin", "wb");
    fwrite(shot_h,sizeof(float),nx*nt, shotgather);
    fclose(shotgather);
    
    FILE *field;
    field=fopen("field.bin", "wb");
    fwrite(P,sizeof(float),nx*nz*nt, field);
    fclose(field);

    
    cudaFree(source);
    cudaFree(vel_O);
    cudaFree(a_x);
    cudaFree(a_z);
    cudaFree(b_x);
    cudaFree(b_z);
    cudaFree(psi_px);
    cudaFree(psi_pz);
    cudaFree(z_px);
    cudaFree(z_pz);
    cudaFree(aten_px);
    cudaFree(aten_pz);
    cudaFree(P_pre);
    cudaFree(P_past);
    cudaFree(shot);
    
    free(source_h);
    free(O_vel_h);
    free(shot_h);
    free(P);
        
    return 0;
}