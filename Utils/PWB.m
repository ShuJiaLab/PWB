%% PWB Deconvolution
parallel.gpu.enableCUDAForwardCompatibility(true)

%% load PSF
 load('..\hPSF\FLF_HyRPSF_-200.000_+200.000_Nor.mat','FLF_HyRPSF_Nor');
 load('..\hPSF\FLFM_PSF_CenfitCoordinates_sub.mat','FLF_ExpPSF_sub_Xfit','FLF_ExpPSF_sub_Yfit');
 
%% load image data
load('..\FLFM_image\FLF_ExpImg_74cCrpnor_0.01-0.01-0.01s_Frm00001.mat','FLFimg_74cCrpnor');

%% crop PSF and images
elementImage_size = 501;
Image_size = [elementImage_size,elementImage_size];
num_cols = 3;
indices_to_crop = [1,2,3,4,5,6,7,8,9];
cropped_data_imgs = cell(1, numel(indices_to_crop));
cropped_psf       = cell(1, numel(indices_to_crop));
FLF_HyRPSF_NorCr = FLF_HyRPSF_Nor(:,:,1:4:1001);

FLF_SubPSF_crp_rad = 250;
FLFPSF_74_NorCrp = S21_DataSave_FLF_HySPSF_Back( FLF_HyRPSF_NorCr ,FLF_SubPSF_crp_rad ,3,...
        FLF_ExpPSF_sub_Yfit,FLF_ExpPSF_sub_Xfit,0);

for k = 1:numel(indices_to_crop)
    idx = indices_to_crop(k);    
    row = ceil(idx / num_cols);
    col = mod(idx-1, num_cols) + 1;   
    x_start = (col-1) * elementImage_size + 1;
    y_start = (row-1) * elementImage_size + 1;   
    for m = 1:1
        cropped_data_imgs{m,k} = FLFimg_74cCrpnor(y_start:y_start+elementImage_size-1, x_start:x_start+elementImage_size-1, :);
    end
    cropped_psf{k}       = FLFPSF_74_NorCrp(y_start:y_start+elementImage_size-1, x_start:x_start+elementImage_size-1, :);
end
disp('PSF cropped');

%% get WB OTF
OTFfp_store = cell(1, numel(indices_to_crop));
OTFbp_WB_store = cell(1, numel(indices_to_crop));

Magnify_val = 65535;

for o = 1:numel(indices_to_crop) % % calculate the OTF
    [OTF1, OTF2_WB] = GetOTF_WB( single( cropped_psf{o} ) );gpuDevice;    
    OTFfp_data_temp  = gpuArray(single( Magnify_val.* OTF1));
    OTFfp_store{o}  = OTFfp_data_temp;   
    OTFbp_WB_data_temp  = gpuArray(single( Magnify_val.* OTF2_WB));
    OTFbp_WB_store{o}  = OTFbp_WB_data_temp;
end
disp('OTF obtained')

%% PWB deconvolution for green channel
for j = 1:1
    % initialization
    tic
    [FFTPad_Tep,Recon_data_green] = FLFM_RecImageSetting(cropped_data_imgs{2}, OTFfp_store{2});
    for i = [2,4,6,8]
        idx = mod(i-1, numel(indices_to_crop)) + 1;
        OTFfptemp = OTFfp_store{1, idx};
        OTFbptemp = OTFbp_WB_store{1, idx};
        patchtemp = single(cropped_data_imgs{j, idx});
        [Recon_proj] = FProj(Image_size, FFTPad_Tep, OTFfptemp, Recon_data_green);
        [Recon_data_green] = BProj(Image_size, FFTPad_Tep, OTFbptemp, Recon_data_green, patchtemp, Recon_proj);
        Recon_data_green(isnan(Recon_data_green)) = 0;
        Recon_data_green = abs(Recon_data_green);
        U65_Imshow_B16_MIP_3D_Fire(Recon_data_green);
        disp(i)
    end
    toc
    A = U65_Imshow_B16_MIP_3D_Col(Recon_data_green,023,1);
end

%% PWB deconvolution for red channel
for j = 1:1
    % initialization
    tic
    [FFTPad_Tep,Recon_data_red] = FLFM_RecImageSetting(cropped_data_imgs{1}, OTFfp_store{1});
    for i = [1,3,5,7,9]
        idx = mod(i-1, numel(indices_to_crop)) + 1; 
        OTFfptemp = OTFfp_store{1, idx}; 
        OTFbptemp = OTFbp_WB_store{1, idx}; 
        patchtemp = single(cropped_data_imgs{j, idx}); 
        [Recon_proj] = FProj(Image_size, FFTPad_Tep, OTFfptemp, Recon_data_red);
        [Recon_data_red] = BProj(Image_size, FFTPad_Tep, OTFbptemp, Recon_data_red, patchtemp, Recon_proj);
        Recon_data_red(isnan(Recon_data_red)) = 0;
        Recon_data_red = abs(Recon_data_red);
        U65_Imshow_B16_MIP_3D_Fire(Recon_data_red);
        disp(i)
    end
    toc
    B = U65_Imshow_B16_MIP_3D_ColAdj(Recon_data_red,022,1);
end

%% Assemble MIP of two channels
C = A + B; 
figure
imshow(C)
