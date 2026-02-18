function [FFTPad_Tep,Recon_data,Image_size] = FLFM_RecImageSetting(FLFMImage, OTF)
    Image_size   = [size(FLFMImage,1), size(FLFMImage,2)];
    PSF__depth   = size(OTF,3);
    Recon_size   = [size(OTF,1), size(OTF,2)];
    Recon_data   = ones([Image_size, PSF__depth], 'single');
    FFTPad_Tep   = gpuArray.ones(Recon_size, 'single');
    disp('---Image setting is done---')
end