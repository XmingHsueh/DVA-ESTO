function L=  LaplacianST(ys,yt)
ns = length(ys);
nt = length(yt);
Wst = zeros(ns,nt);
for i = 1:ns
    for j = 1:nt
        if ys(i)==yt(j)
            Wst(i,j) = 1;
        end
    end
end
L = diag(sum(1/ns/nt*Wst))-1/ns/nt*Wst;