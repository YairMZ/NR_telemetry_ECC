N_Bayes = 1;
p1_vec = 0.01:0.005:0.99;
f_vec = 0:0.005:0.1;

trials = 10000*5;

[p_mesh, f_mesh] = meshgrid(p1_vec,f_vec);
ignorant_bayes_50 = zeros(size(p_mesh));
bayes_mse_50 = zeros(size(p_mesh));
a=1;
b=1;
min_size = 50;
max_size = 75;
for ii= 1:size(p_mesh,1)
    for jj= 1:size(p_mesh,2)
        p1 = p_mesh(ii,jj);
        f = f_mesh(ii,jj);
        v_hat = zeros(1,trials);
        for kk = 1:trials
            x = rand(1) < p1;
            w = rand(1) < f;
            x = mod(x + w,2);
            a = a + x;
            b = b + 1 - x;
            if a+b>max_size
                m = gcd(a,b);
                a = a/m;
                b = b/m;
                if a+b<min_size
                    m = floor(min_size/(a+b));
                    a = a*m;
                    b = b*m;
                end
            end
            v_hat(kk) = (a-1)/(a+b-2);
        end
        ignorant_error = abs(v_hat - p1);
        p1_hat = min(max((v_hat-f)/(1-2*f),0),1);
        ml_error = abs(p1_hat - p1);
        ignorant_bayes_50(ii,jj) = mean(ignorant_error.^2);
        bayes_mse_50(ii,jj) = mean(ml_error.^2);
    end
end
        
%%
f= figure;
ax = axes('Parent',f);
h=surf(p_mesh,f_mesh,ignorant_bayes_50);
set(h, 'edgecolor','interp');
ylabel('$f$','Interpreter','latex');
xlabel('$p_1$','Interpreter','latex');
a = colorbar;
ylabel(a,'$MSE(\hat{p}_1)$','FontSize',16,'Rotation',270,'Interpreter','latex');
title(['Bayes - MSE ignoring noise, window size ',num2str(min_size)])
a.Label.Position(1)=3;

f= figure;
ax = axes('Parent',f);
h=surf(p_mesh,f_mesh,bayes_mse_50);
set(h, 'edgecolor','interp');
ylabel('$f$','Interpreter','latex');
xlabel('$p_1$','Interpreter','latex');
a = colorbar;
ylabel(a,'$MSE(\hat{p}_1)$','FontSize',16,'Rotation',270,'Interpreter','latex');
title(['Bayes - MSE accouting for noise, window size ',num2str(min_size)])
a.Label.Position(1)=3;

%%
N_Bayes = 75;
p1_vec = 0.01:0.005:0.99;
f_vec = 0:0.005:0.1;

ignorant_bayes_50 = [];
trials = 10000;

results = [];

[p_mesh, f_mesh] = meshgrid(p1_vec,f_vec);
ignorant_mse_75 = zeros(size(p_mesh));
ml_mse_75 = zeros(size(p_mesh));

for ii= 1:size(p_mesh,1)
    for jj= 1:size(p_mesh,2)
        p1 = p_mesh(ii,jj);
        f = f_mesh(ii,jj);
        x = rand(N_Bayes,trials) < p1;
        w = rand(N_Bayes,trials) < f;
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
title(['MSE ignoring noise, window size ',num2str(N_Bayes)])
a.Label.Position(1)=3;

f= figure(4);
ax = axes('Parent',f);
h=surf(p_mesh,f_mesh,ml_mse_75);
set(h, 'edgecolor','interp');
ylabel('$f$','Interpreter','latex');
xlabel('$p_1$','Interpreter','latex');
a = colorbar;
ylabel(a,'$MSE(\hat{p}_1)$','FontSize',16,'Rotation',270,'Interpreter','latex');
title(['MSE accouting for noise, window size ',num2str(N_Bayes)])
a.Label.Position(1)=3;

%%
N_Bayes = 100;
p1_vec = 0.01:0.005:0.99;
f_vec = 0:0.005:0.1;

ignorant_bayes_50 = [];
trials = 10000;

results = [];

[p_mesh, f_mesh] = meshgrid(p1_vec,f_vec);
ignorant_mse_100 = zeros(size(p_mesh));
ml_mse_100 = zeros(size(p_mesh));

for ii= 1:size(p_mesh,1)
    for jj= 1:size(p_mesh,2)
        p1 = p_mesh(ii,jj);
        f = f_mesh(ii,jj);
        x = rand(N_Bayes,trials) < p1;
        w = rand(N_Bayes,trials) < f;
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
title(['MSE ignoring noise, window size ',num2str(N_Bayes)])
a.Label.Position(1)=3;

f= figure(6);
ax = axes('Parent',f);
h=surf(p_mesh,f_mesh,ml_mse_100);
set(h, 'edgecolor','interp');
ylabel('$f$','Interpreter','latex');
xlabel('$p_1$','Interpreter','latex');
a = colorbar;
ylabel(a,'$MSE(\hat{p}_1)$','FontSize',16,'Rotation',270,'Interpreter','latex');
title(['MSE accouting for noise, window size ',num2str(N_Bayes)])
a.Label.Position(1)=3;
