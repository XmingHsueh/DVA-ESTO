clc,clear
warning off;
rand('state',30);
randn('state',17);

%% Generate source and target samples
dim = 2;
X_base = [];
X_basetran = [];
r_max = 0.3;
theta_max = 2*pi;
num_samples = 50;
for i = 1:num_samples
    r = r_max*rand;
    theta = theta_max*rand;
    X_base = [X_base;[r*cos(theta) r*sin(theta)]];
    rtran = r_max*rand;
    thetatran = theta_max*rand;
    X_basetran = [X_basetran;[rtran*cos(thetatran) rtran*sin(thetatran)]];
end
X_s = X_base*[1 0.3;0.3 1]+repmat([0.25,0.25],num_samples,1);
X_stran = X_basetran*[1 0.3;0.3 1]+repmat([0.25,0.25],num_samples,1);
X_t = X_base*[1 -0.3;-0.3 1]+repmat([0.65,0.65],num_samples,1);

fit_s = zeros(num_samples,1);
fit_stran = zeros(num_samples,1);
fit_t = zeros(num_samples,1);
for i = 1:num_samples
    fit_s(i) = X_s(i,1)+X_s(i,2);
    fit_stran(i) = X_stran(i,1)+X_stran(i,2);
    fit_t(i) = X_t(i,2)-X_t(i,1);
end

color_start = [255,0,0];
color_end = [0,0,255];
color_delta = color_end-color_start;
color_s = zeros(num_samples,3);
color_stran = zeros(num_samples,3);
color_t = zeros(num_samples,3);
for i = 1:num_samples
    win_s = length(find(fit_s(i)<=fit_s));
    color_s(i,:) = color_start+color_delta*win_s/num_samples;
    win_stran = length(find(fit_stran(i)<=fit_stran));
    color_stran(i,:) = color_start+color_delta*win_stran/num_samples;
    win_t = length(find(fit_t(i)<=fit_t));
    color_t(i,:) = color_start+color_delta*win_t/num_samples;
end

%% Plot the exact fitness scatter and the relaxed rank scatter
fig_width = 1400;
fig_height = 250;
screen_size = get(0,'ScreenSize');
figure1 = figure('color',[1 1 1],'position',[(screen_size(3)-fig_width)/2, (screen_size(4)-...
    fig_height)/2,fig_width, fig_height]);
subplot(1,5,1);
for i = 1:num_samples
    plot(X_s(i,1),X_s(i,2),'o','markersize',7,'markeredgecolor',color_s(i,:)/255,'linewidth',1);hold on;
    plot(X_t(i,1),X_t(i,2),'^','markersize',7,'markeredgecolor',color_t(i,:)/255,'linewidth',1);
end
set(gca,'fontsize',10,'linewidth',1);
xlabel('$x_1$','fontsize',12,'interpret','latex');
ylabel('$x_2$','fontsize',12,'interpret','latex');
hl=legend('Source solutions','Target solutions');
set(hl,'box','off','edgecolor','none','location','best','fontsize',10,'interpret','latex');
title('Source-Target Solutions','interpret','latex','fontsize',12);

X_tran_kae = adaptation_tools(X_s,fit_s,X_t,fit_t,'KAE',X_stran);
subplot(1,5,2);
for i = 1:num_samples
    plot(X_tran_kae(i,1),X_tran_kae(i,2),'o','markersize',7,'markeredgecolor',color_stran(i,:)/255,'linewidth',1);hold on;
    plot(X_t(i,1),X_t(i,2),'^','markersize',7,'markeredgecolor',color_t(i,:)/255,'linewidth',1);
end
set(gca,'fontsize',10,'linewidth',1);
xlabel('$x_1$','fontsize',12,'interpret','latex');
ylabel('$x_2$','fontsize',12,'interpret','latex');
title('KAE Adaptation','interpret','latex','fontsize',12);

X_tran_nda = adaptation_tools(X_s,fit_s,X_t,fit_t,'NDA',X_stran);
subplot(1,5,3);
for i = 1:num_samples
    plot(X_tran_nda(i,1),X_tran_nda(i,2),'o','markersize',7,'markeredgecolor',color_stran(i,:)/255,'linewidth',1);hold on;
    plot(X_t(i,1),X_t(i,2),'^','markersize',7,'markeredgecolor',color_t(i,:)/255,'linewidth',1);
end
set(gca,'fontsize',10,'linewidth',1);
xlabel('$x_1$','fontsize',12,'interpret','latex');
ylabel('$x_2$','fontsize',12,'interpret','latex');
title('NDA Adaptation','interpret','latex','fontsize',12);

X_tran_div = adaptation_tools(X_s,fit_s,X_t,fit_t,'DIV',X_stran);
subplot(1,5,4);
for i = 1:num_samples
    plot(X_tran_div(i,1),X_tran_div(i,2),'o','markersize',7,'markeredgecolor',color_stran(i,:)/255,'linewidth',1);hold on;
    plot(X_t(i,1),X_t(i,2),'^','markersize',7,'markeredgecolor',color_t(i,:)/255,'linewidth',1);
end
set(gca,'fontsize',10,'linewidth',1);
xlabel('$x_1$','fontsize',12,'interpret','latex');
ylabel('$x_2$','fontsize',12,'interpret','latex');
title('Diversity Adaptation','interpret','latex','fontsize',12);

X_tran_con = adaptation_tools(X_s,fit_s,X_t,fit_t,'CON',X_stran);
subplot(1,5,5);
for i = 1:num_samples
    plot(X_tran_con(i,1),X_tran_con(i,2),'o','markersize',7,'markeredgecolor',color_stran(i,:)/255,'linewidth',1);hold on;
    plot(X_t(i,1),X_t(i,2),'^','markersize',7,'markeredgecolor',color_t(i,:)/255,'linewidth',1);
end
set(gca,'fontsize',10,'linewidth',1);
xlabel('$x_1$','fontsize',12,'interpret','latex');
ylabel('$x_2$','fontsize',12,'interpret','latex');
title('Convergence Adaptation','interpret','latex','fontsize',12);