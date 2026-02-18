function [OTF_FP_data,OTF_BP_data] = GetOTF_WB( PSF_FP_data )
% Hao Wu
% The WB part of this function was adopted from 
% "Rapid image deconvolution and multiview fusion for optical microscopy." Nat Biotechnol 38, 1337–1346 (2020). 

alpha = 1;
beta = 1; 
n = 10;
resFlag = 1; 
iRes = [2,2,0];

[Sx, Sy, Sz] = size(PSF_FP_data);
Scx = (Sx+1)/2; 
Scy = (Sy+1)/2;

% pixel size in Fourier domain
px = 1/Sx; py = 1/Sy; 

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    PSF_OT_data = flip(flip(PSF_FP_data,1),2);
    PSF_03_size = [size(PSF_FP_data,1), size(PSF_FP_data,2)];              
    PSF_03depth = size(PSF_FP_data,3);
    PSF_03_Half = floor(PSF_03_size(1:2)/2);                               
    Rec_03_size = PSF_03_size;
    disp(['FFT size is ', num2str(Rec_03_size(1)),'X',num2str(Rec_03_size(2))]); 
    
    Pad_00_Temp = gpuArray.zeros( Rec_03_size             , 'single' );
        
    OTF_FP_data =          zeros([Rec_03_size,PSF_03depth], 'single' );
    OTF_BP_data =          zeros([Rec_03_size,PSF_03depth], 'single' );
    for idxk        = 1:PSF_03depth                                      
        PSF_FP_temp = Pad_00_Temp;           PSF_FP_temp( 1:PSF_03_size(1), 1:PSF_03_size(2) ) = PSF_FP_data(:,:,idxk);
        PSF_OT_temp = Pad_00_Temp;           PSF_OT_temp( 1:PSF_03_size(1), 1:PSF_03_size(2) ) = PSF_OT_data(:,:,idxk);
        
        OTF_FP_Temp = gather( fft2( exindex( PSF_FP_temp, ( PSF_03_Half(1) + ( 1: Rec_03_size(1) ) ),...
                                                          ( PSF_03_Half(2) + ( 1: Rec_03_size(2) ) ), 'circular') ) );

        
        % calculate PSF FWHM
        [FWHMx, FWHMy, FWHMz] = fwhm_PSF(PSF_FP_data(:,:,idxk));
        % set resolution cutoff
        switch(resFlag)
            case 0 % Set resolution as 1/root(2) of PSF_fp FWHM: iSIM case
                resx = FWHMx/2^0.5;resy = FWHMy/2^0.5;resz = FWHMz/2^0.5; 
            case 1 % Set resolution as PSF_fp FWHM
                resx = FWHMx;resy = FWHMy;resz = FWHMz; 
            case 2 % Set resolution based input values
                resx = iRes(1);resy = iRes(2);resz = iRes(3);
            otherwise
                error('Processing terminated, please set resFlag as 0, 1, or 2')  
        end
        
        % frequency cutoff in terms of pixels
        tx = 1/resx/px; ty = 1/resy/py;
        
        % create Wiener filter
        OTF_flip = gather( fft2( exindex( PSF_OT_temp, ( PSF_03_Half(1) + ( 1: Rec_03_size(1) ) ),...
                                                          ( PSF_03_Half(2) + ( 1: Rec_03_size(2) ) ), 'circular') ) );
        OTF_abs = fftshift(abs(OTF_flip));
        OTFmax = max(OTF_abs(:)); % find maximum value and position
        M =  OTFmax(1);
        OTF_abs_norm = OTF_abs/M;
        OTF_flip_norm = OTF_flip/M;
        OTF_Wiener = OTF_flip_norm ./(abs(OTF_flip_norm).^2+alpha);
        % cutoff gain for winer filter
        OTF_Wiener_abs = fftshift(abs(OTF_Wiener));
        tplane = abs(OTF_Wiener_abs); % central slice
        tline = max(tplane,[],2);        
        to1 = max(round(Scx -tx), 1); to2 = min(round(Scx+tx), Sx);
        beta_wienerx = (tline(to1) + tline(to2))/2; % OTF frequency intensity at cutoff:x

        % create Butteworth Filter
        kcx = tx; % width of Butterworth Filter
        kcy = ty; % width of Butterworth Filter
        ee = beta_wienerx/beta^2 - 1;
        mask_temp = zeros(Rec_03_size(1), Rec_03_size(2));
        for i = 1: Sx
            for j = 1: Sy
                for k = 1:1
                    w = ((i-Scx)/kcx)^2 + ((j-Scy)/kcy)^2;
                    mask_temp(i,j) = 1/sqrt(1+ee*w^n); 
                end
            end
        end     
        mask_temp = ifftshift(mask_temp); % Butterworth Filter
        
        % create Wiener-Butteworth Filter
        %disp(size(mask_temp)); disp(size(OTF_Wiener));
        OTF_BP_Temp = mask_temp.*OTF_Wiener;                                     
        OTF_FP_data(:,:,idxk) = gather(OTF_FP_Temp,[],'all');
        OTF_BP_data(:,:,idxk) = gather(OTF_BP_Temp,[],'all');        

        disp(  [num2str(idxk),'|',num2str(PSF_03depth)])
    end
    
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function arr                        = exindex(  arr, varargin)         
        % Sort out arguments
        [exindices, rules, nd, sz] = getinputs(arr, varargin{:});
        consts    = cellfun(@iscell, rules);  % Check for constants, as can be
        constused = any(consts);              % more efficient if there are none

        % Setup for constant padding
        if constused
            tofill = cell(1, nd);
        end

        % Main loop over subscript arguments, transforming them into valid subscripts into arr using the rule for each dimension
        if constused
            for i = 1:nd
                [exindices{i}, tofill{i}] = extend(exindices{i}, rules{i}, sz(i));
            end
        else % no need for information for doing constants
            for i = 1:nd
                exindices{i} = extend(exindices{i}, rules{i}, sz(i));
            end
        end

        % Create the new array by indexing into arr. 
        % If there are no constants, this does the whole job
        arr = arr(exindices{:});

        % Fill areas that need constants
        if constused
            % Get full range of output array indices
            ranges = arrayfun(@(x) {1:x}, size(arr));
            for i = nd:-1:1    % order matters
                if consts(i)
                    ranges{i} = tofill{i};      % don't overwrite original
                    c = rules{i};               % get constant and fill ...
                    arr(ranges{:}) = c{1};      % we've checked c is scalar
                    ranges{i} = ~tofill{i};     % don't overwrite
                end
            end
        end

    end

    function [exindices, rules, nd, sz] = getinputs(arr, varargin)         
    nd = length(varargin);
        if     nd == 0
            error('exindex:missingargs', 'Not enough arguments');
        elseif nd == 1
            exindices = varargin;
            rules = {{0}};
        elseif ~(isnumeric(varargin{2}) || strcmp(varargin{2}, ':'))           % have alternating indices and rule
            nd = nd/2;
            if round(nd) ~= nd
                error('exindex:badnumargs', ...
                    'Odd number of arguments after initial index/rule pair');
            end
            exindices = varargin(1:2:end);
            rules = varargin(2:2:end);
        elseif nd > 2 && ~(isnumeric(varargin{end}) || strcmp(varargin{end}, ':')) % have a general rule at end
            nd = nd - 1;
            exindices = varargin(1:nd);
            [rules{1:nd}] = deal(varargin{end});
        else                                                                   % no rule is specified
            exindices = varargin;
            [rules{1:nd}] = deal({0});
        end

        % Sort out mismatch of apparent array size and number of dimensions indexed
        sz = size(arr);
        ndarr = ndims(arr);
        if nd < ndarr
            if nd == 1 && ndarr == 2
                if sz(1) == 1 && sz(2) > 1                                     % have a row vector
                    exindices = [{1} exindices {1}];
                    rules = [rules rules];                                     % 1st rule doesn't matter
                elseif sz(2) == 1 && sz(1) > 1                                 % have a column vector
                    exindices = [exindices {1}];
                    rules = [rules rules];                                     % 2nd rule doesn't matter
                else
                    error('exindex:wantvector', ...
                        'Only one index but array is not a vector');
                end
            else
                error('exindex:toofewindices', ...
                    'Array has more dimensions than there are index arguments');
            end
            nd = 2;
        elseif nd > ndarr
            % Effective array size
            sz = [sz ones(1, nd-ndarr)];
        end

        colons = strcmp(exindices, ':');
        if any(colons)  % saves a little time
            exindices(colons) = arrayfun(@(x) {1:x}, sz(colons));
        end

        % Check the indices (rules are checked as required in extend)
        checkindex = @(ind) validateattributes(ind, {'numeric'}, {'integer'}, 'exindex', 'index');
        cellfun(checkindex, exindices);
    end

    function [ind, tofill]              = extend(   ind, rule, s )         
        % The core function: maps extended array subscripts into valid input array subscripts.

        if ischar(rule)    % pad with rule
            tofill = [];   % never used
            switch rule
                case 'replicate'
                    ind = min( max(1,ind),  s);
                case 'circular'
                    ind = mod(     ind-1,   s) + 1;
                case 'symmetric'
                    ind = mod(     ind-1, 2*s) + 1;
                    ott = ind > s;
                    ind(ott) = 2*s + 1 - ind(ott);
                otherwise
                    error('exindex:badopt', 'Unknown option');
            end

        elseif iscell(rule) && isscalar(rule)     % pad with constant

            tofill = ind < 1 | ind > s;
            ind(tofill) = 1;

        else
            error('exindex:badconst', 'Expecting string or scalar cell');
        end
    end
    
    function [FWHMx,FWHMy,FWHMz] = fwhm_PSF(PSF, pixelSize, cFlag, fitFlag)
    % Feed back the full width at half maximun of the input PSF
    % fwhm.m and mygaussfit.m are needed
    % cFlag
    %       0: use maximum's position as PSF center position
    %       1: use matrix's center position as PSF center position
    % fitFlag
    %       0: no fitting before calculate FWHM
    %       1: spine fitting before calculate FWHM
    %       2: gaussian fitting before calculate FWHM
    % 
    if(nargin == 1)
        pixelSize = 1;
        cFlag = 0;
        fitFlag = 0;
    end

    if(nargin == 2)
        cFlag = 0;
        fitFlag = 0;
    end

    if(nargin == 3)
        fitFlag = 0;
    end

    % PSF = PSF - mean(PSF(:));
    [Sx,Sy,Sz] = size(PSF);
    if((Sx ==1)||(Sy==1)) % 1D input
        x = 1:max(Sx,Sy);
        x = x';
        y = PSF(:);
        FWHMx = fwhm(x, y);
        FWHMy = 0;
        FWHMz = 0;
        else if(Sz == 1) % 2D input
            if(cFlag)  
                indx = floor((Sx+1)/2);
                indy = floor((Sy+1)/2);
            else
                [~, ind] = max(PSF(:)); % find maximum value and position 
                [indx,indy] = ind2sub([Sx,Sy],ind(1));
            end

            x = 1:Sx;
            x = x';
            y = PSF(:,indy);
            y = y(:);
            if(fitFlag==1)
                xq = 1:0.1:Sx;
                yq = interp1(x, y, xq, 'spline');
                FWHMx = fwhm(xq, yq);
            elseif(fitFlag==2)
                [sig,~,~] = mygaussfit(x,y);
                FWHMx = sig*2.3548;
            else
                FWHMx = fwhm(x, y);
            end


            x = 1:Sy;
            x = x';
            y = PSF(indx,:);
            y = y(:);
            if(fitFlag==1)
                xq = 1:0.1:Sx;
                yq = interp1(x, y, xq, 'spline');
                FWHMy = fwhm(xq, yq);
            elseif(fitFlag==2)
                [sig,~,~] = mygaussfit(x,y);
                FWHMy = sig*2.3548;
            else
                FWHMy = fwhm(x, y);
            end

            FWHMz = 0;
         else % 3D input
             if(cFlag)  
                indx = floor((Sx+1)/2);
                indy = floor((Sy+1)/2);
                indz = floor((Sz+1)/2);
            else
                [~, ind] = max(PSF(:)); % find maximum value and position 
                [indx,indy,indz] = ind2sub([Sx,Sy,Sz],ind(1));
            end


            x = 1:Sx;
            x = x';
            y = PSF(:,indy,indz);
            y = y(:);
            if(fitFlag==1)
                xq = 1:0.1:Sx;
                yq = interp1(x, y, xq, 'spline');
                FWHMx = fwhm(xq, yq);
            elseif(fitFlag==2)
                [sig,~,~] = mygaussfit(x,y);
                FWHMx = sig*2.3548;
            else
                FWHMx = fwhm(x, y);
            end
            x = 1:Sy;
            x = x';
            y = PSF(indx,:,indz);
            y = y(:);
            if(fitFlag==1)
                xq = 1:0.1:Sy;
                yq = interp1(x, y, xq, 'spline');
                FWHMy = fwhm(xq, yq);
            elseif(fitFlag==2)
                [sig,~,~] = mygaussfit(x,y);
                FWHMy = sig*2.3548;
            else
                FWHMy = fwhm(x, y);
            end

            x = 1:Sz;
            x = x';
            y = PSF(indx,indy,:);
            y = y(:);
            if(fitFlag==1)
                xq = 1:0.1:Sz;
                yq = interp1(x, y, xq, 'spline');
                FWHMz = fwhm(xq, yq);
            elseif(fitFlag==2)
                [sig,~,~] = mygaussfit(x,y);
                FWHMz = sig*2.3548;
            else
                FWHMz = fwhm(x, y);
            end
            FWHMz = fwhm(x, y);
        end
    end

    FWHMx = FWHMx*pixelSize;
    FWHMy = FWHMy*pixelSize;
    FWHMz = FWHMz*pixelSize;
    end
end