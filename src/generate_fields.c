#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, const char * argv[]){	
	int nx = 100;
	int nz = 100;
	float vs=1730;
	float vp=3000;
	float p=2500;

	int i;
	int j;

	float P[nx][nz];
	float Vp[nx][nz];
	float Vs[nx][nz];

	// Crear las matrices 
	for(i= 0; i<nx; i++){
		for(j=0;j<nz;j++){
			P[i][j]=p;
		}
	}
	i=0;
	j=0;


	for(i= 0; i<nx; i++){
		for(j=0;j<nz;j++){
			Vp[i][j]=vp;
		}
	}
	i=0;
	j=0;


	for(i= 0; i<nx; i++){
		for(j=0;j<nz;j++){
			Vs[i][j]=vs;
		}
	}

	// Guardar los binarion
	FILE *bou;
    bou=fopen("density.bin", "w+");
    fwrite(P,sizeof(float),nx*nz, bou);
    fclose(bou);

	FILE *v_p;
    v_p=fopen("VelocityP.bin", "w+");
    fwrite(Vp,sizeof(float),nx*nz, v_p);
    fclose(v_p);

	FILE *v_s;
    v_s=fopen("VelocityS.bin", "w+");
    fwrite(Vs,sizeof(float),nx*nz, v_s);
    fclose(v_s);

    return 0;
}

