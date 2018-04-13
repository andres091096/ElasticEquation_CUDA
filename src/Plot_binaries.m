clear;
clc;
close all;

tend=1;
dt=0.002;
nx=100;
nz=100;
nt=ceil(tend/dt);


%% Cargar Campo Y Crear Volumen
field=fopen('field.bin');
Ux_1=fread(field,'float');

Ux=zeros(nx,nz,nt);

for it = 1:nt
    for iz = 1:nz
        for ix = 1:nx-1
            Ux(ix,iz,it)=Ux_1(ix + ((iz-1)*nx) + ((it-1)*nx*nz));
        end
    end
end

%% Visualizar Campo
for i=1:nt-1        
     imagesc(Ux(:,:,i)'); 
     xlabel('x [meters]')
     ylabel('z [meters]')
     pause(0.00001)
end

%% Visualizar Fuente
% figure;
% source_f=fopen('source.bin');
% S=fread(source_f,'float');
% n=0:dt:tend-dt;
% plot(n,S);
% title('Ricker Wavelet');
% xlabel('Time');
% ylabel('Amplitude');



