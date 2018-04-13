#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define I(ix,iz) (ix)+nx*(iz)
# define PI 3.141592653589793


__global__ void propagator_U(float *Ux, float *Uz, float *Txx, float *Txz, float *Tzz, float *P, int nx, int nz, float dt, float dh)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iz = threadIdx.y + blockDim.y * blockIdx.y;

	if (ix > 0 && ix < (nx-1) && iz > 0 && iz < (nz-1))
	{
			Ux[I(ix,iz)] = Ux[I(ix,iz)] + (1/P[I(ix,iz)])*(dt/dh)*(Txx[I(ix,iz)]-Txx[I(ix-1,iz)]+Txz[I(ix,iz)]-Txz[I(ix,iz-1)]);
			Uz[I(ix,iz)] = Uz[I(ix,iz)] + (1/P[I(ix,iz)])*(dt/dh)*(Txz[I(ix+1,iz)]-Txz[I(ix,iz)]+Tzz[I(ix,iz+1)]-Tzz[I(ix,iz)]);   
	}
}


__global__ void propagator_T(float *Ux, float *Uz, float *Txx, float *Txz, float *Tzz, float *Vp, float *Vs,float *P, float *source, int nx, int nz, float dt, float dh, int sx, int sz, int it)
{       
	int ix = threadIdx.x+blockDim.x*blockIdx.x;
	int iz = threadIdx.y+blockDim.y*blockIdx.y;

    if(ix < nx && iz<nz){
	   if(ix > 0 && ix < (nx-1) && iz > 0 && iz < (nz-1))
	   {  
	   	  Txx[I(ix,iz)]=Txx[I(ix,iz)]  +  ((pow(Vp[I(ix,iz)],2)*P[I(ix,iz)])*(dt/dh)*(Ux[I(ix+1,iz)]-Ux[I(ix,iz)]))   +    ((P[I(ix,iz)]*(pow(Vp[I(ix,iz)],2)-2*pow(Vs[I(ix,iz)],2)))*(dt/dh)*(Uz[I(ix,iz)]-Uz[I(ix,iz-1)]));
		  Tzz[I(ix,iz)]=Tzz[I(ix,iz)]  +  ((pow(Vp[I(ix,iz)],2)*P[I(ix,iz)])*(dt/dh)*(Uz[I(ix,iz)]-Uz[I(ix,iz-1)]))   +    ((P[I(iz,iz)]*(pow(Vp[I(ix,iz)],2)-2*pow(Vs[I(ix,iz)],2)))*(dt/dh)*(Ux[I(ix+2,iz)]-Ux[I(ix,iz)]));
		  Txz[I(ix,iz)]=Txz[I(ix,iz)]  +  (pow(Vs[I(ix,iz)],2)*P[I(ix,iz)])*(dt/dh)*(Ux[I(ix,iz+1)]-Ux[I(ix,iz)]+Uz[I(ix,iz)]-Uz[I(ix-1,iz)]);
	   }

        if (ix == (sx-1) && iz == (sz-1)) 
        {
            //Txx[I(ix,iz)] += source[it];
            Tzz[I(ix,iz)] += source[it];
        }
    }
}

int main()
{

	int nx = 100;
	int nz = 100;
	float dh = 20;
	float dt = 0.002;
	float tend=1;
	int nt = ceil(tend/dt);
    int sx = 50, sz = 50    ;
    float f = 4;

    float *P_h = (float*)calloc(nx*nz,sizeof(float));
    float *Vs_h = (float*)calloc(nx*nz,sizeof(float));
    float *Vp_h = (float*)calloc(nx*nz,sizeof(float));
    float *source_h = (float*)calloc(nt,sizeof(float));
    
    float *U =(float*)calloc(nx*nz*nt ,sizeof(float));


    FILE *ro = fopen("density.bin","rb");
    fread(P_h,nx*nz,sizeof(float),ro);
    fclose(ro);

    FILE *v_p = fopen("VelocityP.bin","rb");
    fread(Vp_h,nx*nz,sizeof(float),v_p);
    fclose(v_p);

    FILE *v_s = fopen("VelocityS.bin","rb");
    fread(Vs_h,nx*nz,sizeof(float),v_s);
    fclose(v_s);


    /******* SOURCE *****/
    int it  = 0;
    float t = 0;
    for (it=0;it<nt;it++)
    {
    source_h[it] = -(1.0-2.0*pow(PI*f*(t-(1.0/f)),2))*exp(-pow(PI*f*(t-(1.0/f)),2));
    t += dt;
    }

    /******* CUDA *******/
    float *Ux, *Uz, *Txx, *Txz, *Tzz, *P, *Vs, *Vp;
    float *source;

    cudaMalloc((void **) &Ux, nx*nz*sizeof(float));
    cudaMalloc((void **) &Uz, nx*nz*sizeof(float));
    cudaMalloc((void **) &Txx, nx*nz*sizeof(float));
    cudaMalloc((void **) &Txz, nx*nz*sizeof(float));
    cudaMalloc((void **) &Tzz, nx*nz*sizeof(float));

    cudaMalloc((void **) &P, nx*nz*sizeof(float));
    cudaMalloc((void **) &Vp, nx*nz*sizeof(float));
    cudaMalloc((void **) &Vs, nx*nz*sizeof(float));

    cudaMalloc((void **) &source, nt*sizeof(float));


    cudaMemcpy(P, P_h, nx*nz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Vs, Vs_h, nx*nz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Vp, Vp_h, nx*nz*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(source, source_h, nt*sizeof(float), cudaMemcpyHostToDevice);


    cudaMemset(Ux, 0, nx*nz*sizeof(float));
    cudaMemset(Uz, 0, nx*nz*sizeof(float));
    cudaMemset(Txx, 0, nx*nz*sizeof(float));
    cudaMemset(Txz, 0, nx*nz*sizeof(float));
    cudaMemset(Tzz, 0, nx*nz*sizeof(float));


    dim3 Grid(((nx-1)/32)+1,((nz-1)/32)+1);
    dim3 Block(32,32);

    for(it=0;it<nt;it++)
    {
        propagator_U <<<Grid,Block>>>(Ux, Uz, Txx, Txz, Tzz, P, nx, nz, dt, dh);
    	propagator_T <<<Grid,Block>>>(Ux, Uz, Txx, Txz, Tzz, Vp, Vs, P, source, nx, nz, dt, dh, sx, sz, it);

        cudaMemcpy(U+(nx*nz*it), Ux , nx*nz*sizeof(float), cudaMemcpyDeviceToHost);
    }

    FILE *field_f;
    field_f=fopen("field.bin", "wb");
    fwrite(U,sizeof(float),nx*nz*nt, field_f);
    fclose(field_f);

    FILE *source_f;
    source_f=fopen("source.bin", "wb");
    fwrite(source_h,sizeof(float),nt, source_f);
    fclose(source_f);

    cudaFree(Ux);
    cudaFree(Uz);
    cudaFree(Txx);
    cudaFree(Txz);
    cudaFree(Tzz);
    cudaFree(P);
    cudaFree(Vs);
    cudaFree(Vp);
    cudaFree(source);


    free(P_h);
    free(Vs_h);
    free(Vp_h);
    free(source_h);
    free(U);

    return 0;  
}