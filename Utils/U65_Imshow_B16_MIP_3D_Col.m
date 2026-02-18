function Image_MaxP_16b = U65_Imshow_B16_MIP_3D_Col(Image_Data,Index_Col,Index_Sat)
%Show max projection of image in 3D direction
% Image_Data = FLFRec_96cFlpco1;
% Index_Sat=1

    MaxIntensity = 65535;
   [Colormap_bt16,~,~] = U06_Color_B16(Index_Col);
    
    Image_norm   = rescale(double(Image_Data));
%     Image_norm   = (double(Image_Data));
    Image_size   = size(          Image_norm );
    
    Image_MaxP_x = squeeze(max(Image_norm,[],1))';
    Image_MaxP_y = squeeze(max(Image_norm,[],2));
    Image_MaxP_z = squeeze(max(Image_norm,[],3));
    
    Seam__size = ceil(Image_size(1)/30);
    Maxpr_sizy =  Image_size(3) + Seam__size + Image_size(1);
    Maxpr_sizx =  Image_size(3) + Seam__size + Image_size(2);
    Image_MaxP = ones(Maxpr_sizy,Maxpr_sizx);
    Image_MaxP( 1:Image_size(1)                              ,                              1:Image_size(2))   = Image_MaxP_z;
    Image_MaxP( 1:Image_size(1)                              ,Image_size(2) + Seam__size + (1:Image_size(3)) ) = Image_MaxP_y;
    Image_MaxP((1:Image_size(3)) + Image_size(1) + Seam__size,                              1:Image_size(2)  ) = Image_MaxP_x;
    
    Image_MaxP_16b = uint16( ind2rgb( uint16(Image_MaxP*MaxIntensity*Index_Sat), Colormap_bt16)*MaxIntensity);
    
    Image_MaxP_16b(Image_size(1)+(1:Seam__size),  :               ,:) = MaxIntensity;
    Image_MaxP_16b(    :             ,Image_size(2)+(1:Seam__size),:) = MaxIntensity;
    Image_MaxP_16b(end+1-(1:Image_size(3)),end+1-(1:Image_size(3)),:) = MaxIntensity;
    
     figure;imshow(Image_MaxP_16b)
end


    