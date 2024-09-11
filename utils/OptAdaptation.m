function X_best = OptAdaptation(Xs,Xt,ys,yt)

[ds,ns] = size(Xs);
[dt,nt] = size(Xt);

T_max = 1000;
tol = 1e-6;
alpha = 0.01;

A = Xs*Xs'/ns;
B = Xt*Xt'/nt;
L = LaplacianST(ys,yt);
C = Xs*(alpha*L)*Xt'/ns/nt;
P = [1 0 0 0;0 0 1 0;0 1 0 0;0 0 0 1];

T = 1;
f_old = 100;
num_try = 20;
X_best = zeros(ds);
f_best = inf;
step_size = 1e-1;
for n = 1:num_try
    X = orth(rand(ds));
    while T<T_max
        f = norm(X'*A*X-B,'fro')+trace(X'*C);
        if norm(f-f_old,2)<tol*f
            break;
        end
        F = 4*A*X*X'*A*X-4*A*X*B+C;
        fv = reshape(F,ds^2,1);
        xv = reshape(X,ds^2,1);
        Jk = kron(X'*A'*X,A)+kron(X'*A',A*X)*P+kron(eye(dt),A*X*X'*A)-kron(B',A);
        xv_update = xv-step_size*inv(Jk)*fv;
        X = reshape(xv_update,ds,ds);
        f_old = f;
        T = T+1;
        %     fprintf('AA: f = %.2f\n',f);
    end
    if f_best>f
        X_best = X;
        f_best = f;
    end
end