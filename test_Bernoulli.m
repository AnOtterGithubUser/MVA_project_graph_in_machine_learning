%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Experiment on Bernoulli rewards %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Build a set of arms (Bernoulli)
clear all
clc

Arm1=armBernoulli(0.8);
Arm2=armBernoulli(0.8);
Arm3=armBernoulli(0.1);
Arm4=armBernoulli(0.75);

MAB={Arm1,Arm2,Arm3,Arm4};
mu1 = MAB{1}.mean;
mu2 = MAB{2}.mean;
mu3 = MAB{3}.mean;
mu4 = MAB{4}.mean;
mu = [mu1, mu2, mu3, mu4];


%% Build a decision set
S = [1,1,0,0;1,0,1,0;1,0,0,1;0,1,1,0;0,1,0,1;0,0,1,1]';% Each action plays two arms

%% Initialize players
observed_weights = [MAB{1}.sample(), MAB{2}.sample(), MAB{3}.sample(), MAB{4}.sample()];
eta = 0.001;
M = 20;
beta = 1;
player1 = CombUCB1(S, observed_weights);
player2 = FPL(S, eta, M, beta);

%% Iterate
T = 5000;
actions2 = [];
A2 = [];
for t=1:T
    observed_weights = [MAB{1}.sample(), MAB{2}.sample(), MAB{3}.sample(), MAB{4}.sample()];
    % Play
    player1.play(observed_weights,t);
    [act, v] = player2.play(0);
    actions2 = [actions2,act];
    A2 = [A2,v];
    K = player2.GR(0);
    1-observed_weights'
    player2.UpdateLoss(1-observed_weights');
end

actions1 = player1.action;
A1 = zeros(4, T);
for t=1:T
    A1(:,t) = S(:,actions1(t));
end
%% Regrets on one run 

max_mu = sort(mu, 'descend');
max_mu = max_mu(1) + max_mu(2);
muA1 = mu * A1;
muA2 = mu * A2;
regret1 = max_mu*(1:T) - cumsum(muA1);
regret2 = max_mu*(1:T) - cumsum(muA2);
figure()
plot(regret1,'b');
hold on
plot(regret2,'r');
legend('CombUCB1 player','FPL player');
title('Regrets on one run');

%% Expectation of the regret
nb_simu = 200;
T = 10000;
A_1_exp = zeros(nb_simu,T);
A_2_exp = zeros(nb_simu,T);
for i=1:nb_simu
    i
    observed_weights = [MAB{1}.sample(), MAB{2}.sample(), MAB{3}.sample(), MAB{4}.sample()];
    eta = 0.001;
    M = 20;
    beta = 1;
    player1 = CombUCB1(S, observed_weights);
    player2 = FPL(S, eta, M, beta);
    actions2 = zeros(1,T);
    A2 = zeros(4,T);
    for t=1:T
        observed_weights = [MAB{1}.sample(), MAB{2}.sample(), MAB{3}.sample(), MAB{4}.sample()];
        % Play
        player1.play(observed_weights,t);
        [act, v] = player2.play(0);
        actions2(t) = act;
        A2(:,t) = v;
        K = player2.GR(0);
        player2.UpdateLoss(1-observed_weights');
    end

    actions1 = player1.action;
    A1 = zeros(4, T);
    for t=1:T
        A1(:,t) = S(:,actions1(t));
    end
    A_1_exp(i,:) = cumsum(mu*A1);
    A_2_exp(i,:) = cumsum(mu*A2);
end

%% Plot
A1_reg = mean(A_1_exp);
A2_reg = mean(A_2_exp);
max_mu = sort(mu, 'descend');
max_mu = max_mu(1) + max_mu(2);
figure()
plot(max_mu*(1:T) - A1_reg);
hold on
plot(max_mu*(1:T) - A2_reg);
legend('CombUCB1 player','FPL player');
title('Expected regret (200 runs)');





