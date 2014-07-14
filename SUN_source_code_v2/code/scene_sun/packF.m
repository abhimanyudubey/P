function X = packF(F,featureName,weighted,bow,normalize)

    for i=1:length(F)
        switch featureName                
            case {'geo_texton','geo_color'}
                if bow
                    if normalize
                        X(:,i) = F{i}.hists.all / (sum(F{i}.hists.all)+eps);
                    else
                        X(:,i) = F{i}.hists.all;
                    end
                else
                    if normalize
                        F{i}.hists.gnd = F{i}.hists.gnd / (sum(F{i}.hists.gnd)+eps);
                        F{i}.hists.por = F{i}.hists.por / (sum(F{i}.hists.por)+eps);
                        F{i}.hists.sky = F{i}.hists.sky / (sum(F{i}.hists.sky)+eps);
                        F{i}.hists.vrt = F{i}.hists.vrt / (sum(F{i}.hists.vrt)+eps);
                        F{i}.hists.all = F{i}.hists.all / (sum(F{i}.hists.all)+eps);
                    end
                    if weighted
                        X(:,i) = [F{i}.hists.gnd; F{i}.hists.por; F{i}.hists.sky; F{i}.hists.vrt; 2 * F{i}.hists.all];
                    else
                        X(:,i) = [F{i}.hists.gnd; F{i}.hists.por; F{i}.hists.sky; F{i}.hists.vrt; F{i}.hists.all];
                    end
                end
            case {'denseSIFT','hog2x2','texton','ssim','lbp','lbphf'}               
                if bow
                    if normalize
                        X(:,i) = F{i}.hists.L0/(sum(F{i}.hists.L0)+eps);
                    else
                        X(:,i) = F{i}.hists.L0;
                    end
                else
                    if normalize
                        if weighted
                            X(:,i) = [F{i}.hists.L0/ (sum(F{i}.hists.L0)+eps); 4* F{i}.hists.L1/ (sum(F{i}.hists.L1)+eps); 16* 2*F{i}.hists.L2/ (sum(F{i}.hists.L2)+eps)];
                        else
                            X(:,i) = [F{i}.hists.L0/ (sum(F{i}.hists.L0)+eps); 4* F{i}.hists.L1/ (sum(F{i}.hists.L1)+eps); 16* F{i}.hists.L2/ (sum(F{i}.hists.L2)+eps)];
                        end
                    else
                         if weighted
                            X(:,i) = [F{i}.hists.L0; F{i}.hists.L1; 2*F{i}.hists.L2];
                        else
                            X(:,i) = [F{i}.hists.L0; F{i}.hists.L1; F{i}.hists.L2];
                         end
                    end
                end
            case 'geo_map8x8'
                X(:,i) = F{i}.geo_map8x8;
            case 'tiny_image'
                X(:,i) = F{i}.tiny_image;
            case {'gist','gistPadding'}
                X(:,i) = F{i}.descrs;
            case 'sparse_sift'
                if normalize
                    X(:,i) = [ F{i}.hists.hesaff/(sum(F{i}.hists.hesaff)+eps); F{i}.hists.mser/(sum(F{i}.hists.mser)+eps)];
                else
                    X(:,i) = [ F{i}.hists.hesaff; F{i}.hists.mser];
                end
            case 'line_hists'
                if normalize
                    X(:,i) = [ F{i}.hists.angle/(sum(F{i}.hists.angle)+eps); F{i}.hists.length/(sum(F{i}.hists.length)+eps)];
                else
                    X(:,i) = [ F{i}.hists.angle; F{i}.hists.length];
                end
                
        end
    end