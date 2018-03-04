%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% Shortest path search problem %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Solves a (4)x(4) grid shortest path
clear all
clc

%% Build a graph
s = [1,1,2,2,3,3,4,5,5,6,6,7,7,8,9,9,10,10,11,11,12,13,14,15];
t = [2,5,3,6,4,7,8,6,9,7,10,8,11,12,10,13,11,14,12,15,16,14,15,16];
A = zeros(16,16);
for i=1:length(s)
    A(s(i),t(i)) = 1;
end
G = digraph(s,t);

Edges = {[1,2],[2,3],[3,4],[1,5],[2,6],[3,7],[4,8],[5,6],[6,7],[7,8],[5,9],[6,10],[7,11],[8,12],[9,10],[10,11],[11,12],[9,13],[10,14],[11,15],[12,16],[13,14],[14,15],[15,16]};


%% Build a decision set
S = zeros(24,20);
S([1,2,3,7,14,21],1) = 1;
S([1,2,6,10,14,21],2) = 1;
S([1,2,6,13,17,21],3) = 1;
S([1,2,6,13,20,24],4) = 1;
S([1,5,9,10,14,21],5) = 1;
S([1,5,9,13,17,21],6) = 1;
S([1,5,9,13,20,24],7) = 1;
S([1,5,12,16,17,21],8) = 1;
S([1,5,12,16,20,24],9) = 1;
S([1,5,12,19,23,24],10) = 1;
S([4,8,9,13,17,21],11) = 1;
S([4,8,9,13,20,24],12) = 1;
S([4,8,9,10,14,21],13) = 1;
S([4,8,12,16,17,21],14) = 1;
S([4,8,12,16,20,24],15) = 1;
S([4,8,12,19,23,24],16) = 1;
S([4,11,15,16,17,21],17) = 1;
S([4,11,15,16,20,24],18) = 1;
S([4,11,15,19,23,24],19) = 1;
S([4,11,18,22,23,24],20) = 1;

%% Build a Gaussian arm for each edge
Arm1 = armGauss(0.95,0.1);
Arm2 = armGauss(0.8,0.1);
Arm3 = armGauss(0.8,0.1);
Arm4 = armGauss(0.8,0.1);
Arm5 = armGauss(0.3,0.1);
Arm6 = armGauss(0.2,0.1);
Arm7 = armGauss(0.8,0.1);
Arm8 = armGauss(0.6,0.1);
Arm9 = armGauss(0.6,0.1);
Arm10 = armGauss(0.6,0.1);
Arm11 = armGauss(0.8,0.1);
Arm12 = armGauss(0.1,0.1);
Arm13 = armGauss(0.1,0.1);
Arm14 = armGauss(0.8,0.1);
Arm15 = armGauss(0.5,0.1);
Arm16 = armGauss(0.5,0.1);
Arm17 = armGauss(0.5,0.1);
Arm18 = armGauss(0.8,0.1);
Arm19 = armGauss(0.2,0.1);
Arm20 = armGauss(0.6,0.1);
Arm21 = armGauss(0.8,0.1);
Arm22 = armGauss(0.8,0.1);
Arm23 = armGauss(0.8,0.1);
Arm24 = armGauss(0.1,0.1);
MAB={Arm1,Arm2,Arm3,Arm4,Arm5,Arm6,Arm7,Arm8,Arm9,Arm10,Arm11,Arm12,Arm13,Arm14,Arm15,Arm16,Arm17,Arm18,Arm19,Arm20,Arm21,Arm22,Arm23,Arm24};
mu = [];
for i=1:24
    mu = [mu; MAB{i}.mean];
end

%% Initialize two players
observed_weights = zeros(1,24);
for i=1:24
    observed_weights(i) = MAB{i}.sample();
end
eta = 0.0032;
M = 11;
beta = 1;
player1 = CombUCB1(S, observed_weights);
player2 = FPL(S, eta, M, beta);

%% Iterate
T = 5000;
actions2 = [];
A2 = [];
for t=1:T
    observed_weights = zeros(1,24);
    for i=1:24
        observed_weights(i) = MAB{i}.sample();
    end
    % Play
    player1.play(observed_weights,t);
    [act, v] = player2.play(0);
    actions2 = [actions2,act];
    A2 = [A2,v];
    K = player2.GR(0);
    player2.UpdateLoss(1-observed_weights');
end

actions1 = player1.action;
A1 = zeros(24, T);
for t=1:T
    A1(:,t) = S(:,actions1(t));
end
disp('Run finished');
%% Find best hindsight action
rewards = mu;
hindsight_action = 1;
max_reward = S(:,hindsight_action)' * rewards;
for i=1:size(S,2)
    if(S(:,i)'*rewards > max_reward)
        max_reward = S(:,i)'*rewards;
        hindsight_action = i;
    end
end
p = plot(G);
p.XData = [1,3,5,7,1,3,5,7,1,3,5,7,1,3,5,7];
p.YData = [7,7,7,7,5,5,5,5,3,3,3,3,1,1,1,1];
% highlight the best path
best_path = zeros(2,6);
compt = 1;
for i=1:24
    if(S(i,hindsight_action) == 1)
        best_path(1,compt) = Edges{i}(1);
        best_path(2,compt) = Edges{i}(2);
        compt = compt+1;
    end
end
highlight(p,best_path(1,:),best_path(2,:),'EdgeColor','r','LineWidth',3);

%% Plot regrets on one run
T = 5000;
rewards = repmat(mu,[1,T]);
rA1 = diag(A1' * rewards);
rA2 = diag(A2' * rewards);  
regret1 = max_reward*(1:T) - cumsum(rA1');
regret2 = max_reward*(1:T) - cumsum(rA2');

figure()
plot(regret1,'b');
hold on
plot(regret2,'r');
legend('CombUCB1 player','FPL player');
title('Regrets on one run');

%% Plot evolution

for t=1:T
    % path 1
    path1 = zeros(2,6);
    compt = 1;
    for i=1:24
        if(S(i,actions1(t)) == 1)
            path1(1,compt) = Edges{i}(1);
            path1(2,compt) = Edges{i}(2);
            compt = compt+1;
        end
    end
    % path 2
    path2 = zeros(2,6);
    compt = 1;
    for i=1:24
        if(S(i,actions2(t)) == 1)
            path2(1,compt) = Edges{i}(1);
            path2(2,compt) = Edges{i}(2);
            compt = compt+1;
        end
    end
    % plot
    subplot(2,2,1)
    p = plot(G);
    p.XData = [1,3,5,7,1,3,5,7,1,3,5,7,1,3,5,7];
    p.YData = [7,7,7,7,5,5,5,5,3,3,3,3,1,1,1,1];
    highlight(p,path1(1,:),path1(2,:),'EdgeColor','r','LineWidth',3);
    title(sprintf('CUCB, Iteration %0.4d, reward: %.2f',[t,A1(:,t)'*mu]));
    subplot(2,2,2)
    plot(regret1(1:t),'b');
    xlim([0, 2500])
    ylim([0, 800])
    subplot(2,2,3)
    p = plot(G);
    p.XData = [1,3,5,7,1,3,5,7,1,3,5,7,1,3,5,7];
    p.YData = [7,7,7,7,5,5,5,5,3,3,3,3,1,1,1,1];
    highlight(p,path2(1,:),path2(2,:),'EdgeColor','r','LineWidth',3);
    title(sprintf('FPL, Iteration %0.4d, reward: %.2f',[t,A2(:,t)'*mu]));
    subplot(2,2,4)
    plot(regret2(1:t),'r');
    xlim([0, 2500]);
    ylim([0,800]);
    print(sprintf('../path/path%d',t),'-dpng');
end

%% Make a video
workingDir = '../path/';
imageNames = dir(fullfile(workingDir,'*.png'));
imageNames = {imageNames.name}';
outputVideo = VideoWriter(fullfile(workingDir,'CUCB_search.avi'));
outputVideo.FrameRate = 5;
open(outputVideo)
for ii = 5:(length(imageNames)-5)
    ii
    img = imread(sprintf('../path/path%d.png',ii));
    writeVideo(outputVideo,img)
end
close(outputVideo)

%% Expected regret
figure()
nb_simu = 200;
T = 10000;
A1_exp = [];
A2_exp = [];
rewards = repmat(mu,[1,T]);
for i=1:nb_simu
    i
    % reset the players
    observed_weights = zeros(1,24);
    for j=1:24
        observed_weights(j) = MAB{j}.sample();
    end
    eta = 0.001;
    M = 20;
    beta = 1;
    player1 = CombUCB1(S, observed_weights);
    player2 = FPL(S, eta, M, beta);
    % play
    actions2 = [];
    A2 = [];
    for t=1:T
        observed_weights = zeros(1,24);
        for i=1:24
            observed_weights(i) = MAB{i}.sample();
        end
        % Play
        player1.play(observed_weights,t);
        [act, v] = player2.play(0);
        actions2 = [actions2,act];
        A2 = [A2,v];
        K = player2.GR(0);
        player2.UpdateLoss(1-observed_weights');
    end

    actions1 = player1.action;
    A1 = zeros(24, T);
    for t=1:T
        A1(:,t) = S(:,actions1(t));
    end
    
    
    rA1 = diag(A1' * rewards);
    rA2 = diag(A2' * rewards);
    A1_exp = [A1_exp;max_reward*(1:T) - cumsum(rA1')];
    A2_exp = [A2_exp;max_reward*(1:T) - cumsum(rA2')];
    plot(max_reward*(1:T) - cumsum(rA1'),'b')
    hold on
    plot(max_reward*(1:T) - cumsum(rA2'),'r');
end

%% Compute expected regret
A1_reg = mean(A1_exp);
A2_reg = mean(A2_exp);
exp_regret1 = A1_reg;
exp_regret2 = A2_reg;

%% Plot expected regret

figure()
plot(exp_regret1,'b');
hold on
plot(exp_regret2,'r');
title('Expected regret (200 runs)');


