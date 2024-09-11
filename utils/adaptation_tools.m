function X_trans = adaptation_tools(Xs,ys,Xt,yt,method,Xtran)

num_samples = length(ys);
fit_s = ys;
fit_t = yt;
medians = median(fit_s);
mediant = median(fit_t);
inds_good = fit_s<=medians;
inds_bad = fit_s>medians;
indt_good = fit_t<=mediant;
indt_bad = fit_t>mediant;
num_classes = 2;
fit_s_rank = fit_relax(fit_s,num_classes);
fit_t_rank = fit_relax(fit_t,num_classes);

switch method
    case 'AE'
        [~,inds] = sort(ys);
        [~,indt] = sort(yt);
        A = Xs(inds,:);
        B = Xt(indt,:);
        X_ae = [A ones(num_samples,1)];
        Y_ae = B;
        M_ae = (X_ae'*X_ae)\X_ae'*Y_ae;
        X_trans = [Xtran ones(num_samples,1)]*M_ae;
    case 'KAE'
        [~,inds] = sort(ys);
        [~,indt] = sort(yt);
        A = Xs(inds,:);
        B = Xt(indt,:);
        source_kernel = kernel_cal(A,A);
        Mk = source_kernel\B;
        Transfer_Kernel = kernel_cal(Xtran,A);
        X_trans = Transfer_Kernel*Mk;
    case 'DIV'
        ms = mean(Xs);
        Cs = cov(Xs);
        Ls = chol(inv(Cs+1e-3*eye(2)));
        mt = mean(Xt);
        Ct = cov(Xt);
        Lt = chol(inv(Ct));
        A = inv(Lt')*Ls;
        b = mt'-A*ms';
        X_trans = [];
        for i =1:num_samples
            X_trans = [X_trans;transpose(A*Xtran'+b)];
        end
    case 'CON'
        [X_sn_total,~,~] = zscore([Xs;Xtran]);
        X_sn = X_sn_total(1:num_samples,:);
        X_trann = X_sn_total(num_samples+1:end,:);
        [X_tn,mu_t,sigma_t] = zscore(Xt);
        Pa = OptAdaptation(X_sn',X_tn',fit_s_rank,fit_t_rank);
        X_tran_aa_n = transpose(Pa'*X_trann');
        X_trans = zeros(num_samples,2);
        for i = 1:num_samples
            X_trans(i,:) = X_tran_aa_n(i,:).*sigma_t+mu_t;
        end
    case 'NDA'
        [~,inds] = sort(ys);
        [~,indt] = sort(yt);
        f_activate=@(x)1./(1+exp(-x));
        num_hiddens = 4;
        source_inputs = [Xs(inds,:) ones(num_samples,1)];
        target_inputs = Xt(indt,:);
        W_ih = rand(size(source_inputs,2),num_hiddens);
        H = f_activate(source_inputs*W_ih);
        W_ho = H\target_inputs;
        f_mapping = @(x)f_activate(x*W_ih)*W_ho;
        X_trans = f_mapping([Xtran ones(size(Xtran,1),1)]);
end