function [Recon_Proj] = FProj(Image_size,FFTPad_Tep,OTF,Recon_data)
% 3D ---> 2D
Recon_Proj = gpuArray.zeros( Image_size             ,'single');
for idxk = 1:size(Recon_data,3)
    PrjFFT_Pad = FFTPad_Tep;
    PrjFFT_Pad( 1:Image_size(1), 1:Image_size(2) ) = Recon_data(:,:,idxk);
    Prj_SLayer =        real( ifft2( fft2(PrjFFT_Pad).* OTF(:,:,idxk) ) ) ;% Inv FT
    Recon_Proj = Recon_Proj + Prj_SLayer( 1:Image_size(1), 1:Image_size(2));
end
end