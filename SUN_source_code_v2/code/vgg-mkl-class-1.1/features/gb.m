function feat = gb(conf, I)

if size(I,3) > 1
  I = rgb2gray(I) ;
end

if ~ conf.usePBEdges
  fbr = compute_channels_oe_nms(I) ;
else
  feat = pbEdges(conf, I) ;
  th  = 4 * mod(feat.theta, pi) / pi ;
  i = floor(th) ;
  wi = 1 - (th - i) ;
  j = i + 1 ;
  wj = 1 - wi ;
  i = mod(i,4) + 1 ;
  j = mod(j,4) + 1 ;

  fbr = zeros(size(I,1),size(I,2),4) ;

  fbr=vl_binsum(fbr,wi .* feat.pb,i,3) ;
  fbr=vl_binsum(fbr,wj .* feat.pb,j,3) ;
end

[descrs, frames] = get_descriptors(...
  fbr, ...
  conf.numFeats, ...
  conf.repulsionRadius, ...
  conf.blurRate, ...
  conf.blurBase, ...
  conf.sampleRadii, ...
  conf.numSamplePerRadius) ;

frames = flipud(frames') ;
descrs = [ descrs(:,:,1) descrs(:,:,2) descrs(:,:,3) descrs(:,:,4)]' ;

feat.frames = frames ;
feat.descrs = descrs ;
