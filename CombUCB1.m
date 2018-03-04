classdef CombUCB1 < handle
    % This class is a CombUCB1 player for a combinatorial semi bandit
    % problem
    
    properties
        S; % Set of possible actions (number of arms x number of actions)
        N; % Number of arms
        w; % Estimated weights of the arms (1 x number of arms)
        T; % Number of times each arm was played (1 x number of arms)
        action; % Keep track of the actions played
    end
    
    methods
        function self = CombUCB1(S, observed_weights)
            % S is the set of possible actions
            % w is a pull on each arm
            self.S = S;
            self.N = size(S,1);
            self.T = ones(1,self.N);
            self.action = [];
            % Initialization
            self.w = zeros(1, self.N);
            u = ones(1, self.N);
            t = 1;
            while (sum(u) > 0) % While there exists one arm for which u is still 1
                [o, A_best] = oracle(self.S, u'); % Choose best arm
                for i=1:length(A_best)
                    if(A_best(i) ~= 0) % if the arm is played by A_best
                        self.w(i) = observed_weights(i);
                        u(i) = 0;
                    end
                end
                t = t+1;
            end
        end
        
        function self = play(self, observed_weights, t)
            %%% Compute UCB
            U = zeros(1, self.N);
            for i=1:self.N % for all arms
                U(i) = self.w(i) + sqrt((1.5*log(t))/self.T(i));
            end
            %%% Pick an arm
            [o, A_t] = oracle(self.S,U');
            self.action = [self.action, o];
            %%% Observe the weights and update
            % Update the statistics
            for i=1:length(A_t)
                if(A_t(i) ~= 0) % if the arm is played by A_t
                    self.T(i) = self.T(i) + 1;
                end
            end
            % Update the estimates of the observed weights
            for i=1:length(A_t)
                if(A_t(i) ~= 0) % if the arm is played by A_t
                    self.w(i) = ((self.T(i)-1) * self.w(i) + observed_weights(i))/self.T(i);
                end
            end
        end
        
    end
end

