%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Experiment on Random Walk rewards %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Build a set of arms (Random Walk)
clear all
clc

Arm1 = armRW(0.8, 0.01);
Arm2 = armRW(0.1, 0.01);
Arm3 = armRW(0.3, 0.01);
Arm4 = armRW(0.8, 0.01);

MAB={Arm1,Arm2,Arm3,Arm4}; 

%% Build a decision set
S = [1,1,0,0;1,0,1,0;1,0,0,1;0,1,1,0;0,1,0,1;0,0,1,1]';% Each action plays two arms

%% Initialize players
observed_weights = [MAB{1}.sample(), MAB{2}.sample(), MAB{3}.sample(), MAB{4}.sample()];
eta = 0.0065;
M = 15;
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
    player2.UpdateLoss(1-observed_weights');
end

actions1 = player1.action;
A1 = zeros(4, T);
for t=1:T
    A1(:,t) = S(:,actions1(t));
end

%% Regrets on one run 

rewards = [MAB{1}.walk; MAB{2}.walk; MAB{3}.walk; MAB{4}.walk];
rewards = rewards(:,1:T);
hindsight_decision = 1;
max_reward = cumsum(S(:,hindsight_decision)' * rewards) ;
for i=2:6
    if(cumsum(S(:,i)' * rewards) > max_reward)
        max_reward = cumsum(S(:,i)' * rewards);
        hindsight_decision = i;
    end
end
rA1 = diag(A1' * rewards);
rA2 = diag(A2' * rewards);
% Alternative
max_reward = S'*rewards;
max_reward = cumsum(max(max_reward));

regret1 = max_reward - cumsum(rA1');
regret2 = max_reward - cumsum(rA2');
figure()
plot(regret1,'b');
hold on
plot(regret2,'r');
legend('CombUCB1 player','FPL player');
title('Regrets on one run');

%% Expectation of the regret
nb_simu = 200; 
T = 5000;
A_1_exp = zeros(nb_simu,T);
A_2_exp = zeros(nb_simu,T);
for i=1:nb_simu
    i
    
    actions2 = [];
    A2 = [];
    Arm1 = armRW(0.8, 0.001);
    Arm2 = armRW(0.1, 0.001);
    Arm3 = armRW(0.3, 0.001);
    Arm4 = armRW(0.8, 0.001);

    MAB={Arm1,Arm2,Arm3,Arm4};
    % Reset the arms
    observed_weights = [MAB{1}.sample(), MAB{2}.sample(), MAB{3}.sample(), MAB{4}.sample()];
    eta = 0.0065;
    M = 17;
    beta = 1;
    player1 = CombUCB1(S, observed_weights);
    player2 = FPL(S, eta, M, beta);
    for t=1:T
        observed_weights = [MAB{1}.sample(), MAB{2}.sample(), MAB{3}.sample(), MAB{4}.sample()];
        % Play
        player1.play(observed_weights,t);
        [act, v] = player2.play(0);
        actions2 = [actions2,act];
        A2 = [A2,v];
        K = player2.GR(0);
        player2.UpdateLoss(1-observed_weights');
    end

    actions1 = player1.action;
    A1 = zeros(4, T);
    for t=1:T
        A1(:,t) = S(:,actions1(t));
    end
    rewards = [MAB{1}.walk; MAB{2}.walk; MAB{3}.walk; MAB{4}.walk];
    rewards = rewards(:,1:T);
%     hindsight_decision = 1;
%     max_reward = cumsum(S(:,hindsight_decision)' * rewards) ;
%     for j=2:6
%         if(cumsum(S(:,j)' * rewards) > max_reward)
%             max_reward = cumsum(S(:,j)' * rewards);
%             hindsight_decision = j;
%         end
%     end
    rA1 = diag(A1' * rewards);
    rA2 = diag(A2' * rewards);
    max_reward = S'*rewards;
    max_reward = cumsum(max(max_reward));
    A_1_exp(i,:) = max_reward - cumsum(rA1');
    A_2_exp(i,:) = max_reward - cumsum(rA2');
end

%% Plot
A1_reg = mean(A_1_exp);
A2_reg = mean(A_2_exp);
figure()
plot(A1_reg);
hold on
plot(A2_reg);
legend('CombUCB1 player','FPL player');
title('Expected regret (200 runs)');
disp('End');