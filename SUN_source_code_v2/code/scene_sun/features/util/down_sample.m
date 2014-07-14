function small=down_sample(big,nr,nc)
% function small=down_sample(big,nr,nc)
% averaging over non-overlapping spatial blocks

cols=fix(linspace(0,size(big,2),nc+1));
rows=fix(linspace(0,size(big,1),nr+1));

small = zeros(nr, nc, size(big,3));
for r=1:nr
  for c=1:nc
    v = big(rows(r)+1:rows(r+1), cols(c)+1:cols(c+1), :);
    v = mean(mean(v,1),2);
    small(r,c,:) = v(:);
  end
end
