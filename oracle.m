function [o, A_best] = oracle(S, w)
% Computes the solution to the problem argmax f(A,w)
% S: set of possible actions (number of arms x number of actions)
% w: vector of weights (number of arms x 1)

o = 1;
A_best = S(:,1);
best_f = f(A_best,w);
for i=1:size(S,2)
    A = S(:,i); % Action
    r = f(A,w);
    if(r > best_f)
        o = i;
        A_best = S(:,i);
        best_f = r;
    end
end
end

