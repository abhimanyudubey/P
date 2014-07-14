function feature = lbp_hf(I)

mapping=getmaplbphf(8);
h=lbp(I,1,8,mapping,'h');
h=h/sum(h);
histograms(1,:)=h;
feature=constructhf(histograms,mapping);