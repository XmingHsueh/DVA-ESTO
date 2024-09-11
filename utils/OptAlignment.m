function [Ps,Pt] = OptAlignment(Xs,Xt,ys,yt,d_low)

[ds,ns] = size(Xs);
[dt,nt] = size(Xt);
T_max = 1000;
tol = 1e-9;
alpha = 0.7;

X_total = [Xs zeros(ds,nt);zeros(dt,ns) Xt];
A = [Xs*Xs'/ns zeros(ds,dt);zeros(dt,ds) -Xt*Xt'/nt];
[Ls,Ld] = LaplacianMatrix(ys,yt);
B = X_total*(alpha*Ls)*X_total';
[Vb,Db] = eig(B);
[~,indb] = sort(diag(Db));
P = Vb(:,indb(1:d_low));

T = 1;
f_old = 100;
while T<T_max
    f = norm(P'*A*P,'fro')+trace(P'*B*P);
    if norm(f-f_old,2)<tol*f
        break;
    end
    M = A*P*P'*A+1/2*B;
    [Vm,Dm] = eig(M);
    [~,indm] = sort(diag(Dm));
    f_old = f;
    P = Vm(:,indm(1:d_low));
    T = T+1;
end
Ps = P(1:ds,:);
Pt = P(ds+1:end,:);