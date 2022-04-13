clear
clc

N_MLE = 50;
p1_vec = 0.01:0.005:0.99;
f_vec = 0:0.005:0.1;

ignorant_mse_50 = [];
trials = 10000;

results = [];

[p_mesh, f_mesh] = meshgrid(p1_vec,f_vec);
ignorant_mse_50 = zeros(size(p_mesh));
ml_mse_50 = zeros(size(p_mesh));

for ii= 1:size(p_mesh,1)
    for jj= 1:size(p_mesh,2)
        p1 = p_mesh(ii,jj);
        f = f_mesh(ii,jj);
        x = rand(N_MLE,trials) < p1;
        w = rand(N_MLE,trials) < f;
        x = mod(x + w,2);
        v_hat = mean(x);
        ignorant_error = abs(v_hat - p1);
        p1_hat = min(max((v_hat-f)/(1-2*f),0),1);
        ml_error = abs(p1_hat - p1);
        ignorant_mse_50(ii,jj) = mean(ignorant_error.^2);
        ml_mse_50(ii,jj) = mean(ml_error.^2);
    end
end
        
%%
f= figure(1);
ax = axes('Parent',f);
h=surf(p_mesh,f_mesh,ignorant_mse_50);
set(h, 'edgecolor','interp');
ylabel('$f$','Interpreter','latex');
xlabel('$p_1$','Interpreter','latex');
a = colorbar;
ylabel(a,'$MSE(\hat{p}_1)$','FontSize',16,'Rotation',270,'Interpreter','latex');
title(['MSE ignoring noise, window size ',num2str(N_MLE)])
a.Label.Position(1)=3;

f= figure(2);
ax = axes('Parent',f);
h=surf(p_mesh,f_mesh,ml_mse_50);
set(h, 'edgecolor','interp');
ylabel('$f$','Interpreter','latex');
xlabel('$p_1$','Interpreter','latex');
a = colorbar;
ylabel(a,'$MSE(\hat{p}_1)$','FontSize',16,'Rotation',270,'Interpreter','latex');
title(['MSE accouting for noise, window size ',num2str(N_MLE)])
a.Label.Position(1)=3;

%%
N_MLE = 75;
p1_vec = 0.01:0.005:0.99;
f_vec = 0:0.005:0.1;

ignorant_mse_50 = [];
trials = 10000;

results = [];

[p_mesh, f_mesh] = meshgrid(p1_vec,f_vec);
ignorant_mse_75 = zeros(size(p_mesh));
ml_mse_75 = zeros(size(p_mesh));

for ii= 1:size(p_mesh,1)
    for jj= 1:size(p_mesh,2)
        p1 = p_mesh(ii,jj);
        f = f_mesh(ii,jj);
        x = rand(N_MLE,trials) < p1;
        w = rand(N_MLE,trials) < f;
        x = mod(x + w,2);
        v_hat = mean(x);
        ignorant_error = abs(v_hat - p1);
        p1_hat = min(max((v_hat-f)/(1-2*f),0),1);
        ml_error = abs(p1_hat - p1);
        ignorant_mse_75(ii,jj) = mean(ignorant_error.^2);
        ml_mse_75(ii,jj) = mean(ml_error.^2);
    end
end
        
%%
f= figure(3);
ax = axes('Parent',f);
h=surf(p_mesh,f_mesh,ignorant_mse_75);
set(h, 'edgecolor','interp');
ylabel('$f$','Interpreter','latex');
xlabel('$p_1$','Interpreter','latex');
a = colorbar;
ylabel(a,'$MSE(\hat{p}_1)$','FontSize',16,'Rotation',270,'Interpreter','latex');
title(['MSE ignoring noise, window size ',num2str(N_MLE)])
a.Label.Position(1)=3;

f= figure(4);
ax = axes('Parent',f);
h=surf(p_mesh,f_mesh,ml_mse_75);
set(h, 'edgecolor','interp');
ylabel('$f$','Interpreter','latex');
xlabel('$p_1$','Interpreter','latex');
a = colorbar;
ylabel(a,'$MSE(\hat{p}_1)$','FontSize',16,'Rotation',270,'Interpreter','latex');
title(['MSE accouting for noise, window size ',num2str(N_MLE)])
a.Label.Position(1)=3;

%%
N_MLE = 100;
p1_vec = 0.01:0.005:0.99;
f_vec = 0:0.005:0.1;

ignorant_mse_50 = [];
trials = 10000;

results = [];

[p_mesh, f_mesh] = meshgrid(p1_vec,f_vec);
ignorant_mse_100 = zeros(size(p_mesh));
ml_mse_100 = zeros(size(p_mesh));

for ii= 1:size(p_mesh,1)
    for jj= 1:size(p_mesh,2)
        p1 = p_mesh(ii,jj);
        f = f_mesh(ii,jj);
        x = rand(N_MLE,trials) < p1;
        w = rand(N_MLE,trials) < f;
        x = mod(x + w,2);
        v_hat = mean(x);
        ignorant_error = abs(v_hat - p1);
        p1_hat = min(max((v_hat-f)/(1-2*f),0),1);
        ml_error = abs(p1_hat - p1);
        ignorant_mse_100(ii,jj) = mean(ignorant_error.^2);
        ml_mse_100(ii,jj) = mean(ml_error.^2);
    end
end
        
%%
f= figure(5);
ax = axes('Parent',f);
h=surf(p_mesh,f_mesh,ignorant_mse_100);
set(h, 'edgecolor','interp');
ylabel('$f$','Interpreter','latex');
xlabel('$p_1$','Interpreter','latex');
a = colorbar;
ylabel(a,'$MSE(\hat{p}_1)$','FontSize',16,'Rotation',270,'Interpreter','latex');
title(['MSE ignoring noise, window size ',num2str(N_MLE)])
a.Label.Position(1)=3;

f= figure(6);
ax = axes('Parent',f);
h=surf(p_mesh,f_mesh,ml_mse_100);
set(h, 'edgecolor','interp');
ylabel('$f$','Interpreter','latex');
xlabel('$p_1$','Interpreter','latex');
a = colorbar;
ylabel(a,'$MSE(\hat{p}_1)$','FontSize',16,'Rotation',270,'Interpreter','latex');
title(['MSE accouting for noise, window size ',num2str(N_MLE)])
a.Label.Position(1)=3;
