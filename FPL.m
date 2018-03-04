classdef FPL < handle
    % This class implements algorithms for learning on combinatorial 
    % optimization with semi bandit feedback
    % among which FPL+GR and its variant FPL+GR.P
    properties
        S;
        eta;
        M;
        L;
        d;
        N;
        action;
        K;
        V;
        L_tilde;
        beta;
        actions;
    end
    
    methods
        function self = FPL(S_, eta_, M_, beta_)
            % Constructor of the class
            self.S = S_;% S is a matrix 
            self.eta = eta_;
            self.M = M_;
            [self.d, self.N] = size(self.S);
            self.beta = beta_;
            self.L = zeros(self.d,1);
            self.L_tilde = zeros(self.d,1);
            self.actions = [];
        end
        
        % The methods FPL, GR and UpdateLoss must be called in that order
        % they implement the algorithm FPL+GR as described in
        % "Importance weighting without importance weights: an efficient
        % algorithm for Combinatorial Semi-Bandits" 2016
        % P is a binary variable
        % P==0 --> FPL+GR
        % P==1 --> FPL+GR.P
        % 1. Call FPL to choose an action V
        % 2. Call GR to get an estimate K 
        % 3. Call UpdateLoss to update the total loss
        function [action, v] = play(self, P)
            % action: index of the action in S
            % v: vector of the action
            L_estimate = 0;
            if(P==1)
                L_estimate = self.L_tilde;
            else
                L_estimate = self.L;
            end
            % Chooses an action within a round using
            
            % Draw Z perturbation with independent components Zi~exp(1)
            Z = exprnd(1,self.d,1);
            
            % Follow the perturbed leader
            %loss_wt = self.eta * L_estimate
            %perturbed_loss = self.eta * L_estimate - Z
            action = 1;
            min_loss = self.S(:,action)' * (self.eta * L_estimate - Z);
            v = self.S(:,action);
            for i=2:self.N
                % Find the action that minimizes the perturbed loss
                if(self.S(:,i)' * (self.eta * L_estimate - Z) < min_loss)
                    action = i;
                    min_loss = self.S(:,action)' * (self.eta * L_estimate - Z);
                    v = self.S(:,action);
                end
            end
            self.action = action;
            self.V = v;
            self.actions = [self.actions, action];
        end
        
        function K = GR(self, P)
            % Geometric resampling
            L_estimate = 0;
            if(P==1)
                L_estimate = self.L_tilde;
            else
                L_estimate = self.L;
            end
            % Initialize waiting list and counter
            K = 0;
            r = self.V;
            for k=1:self.M
                K = K+r;
                % Draw Z' perturbation with independent components
                % Zi~exp(1)
                Z_prime = exprnd(1,self.d,1);
                % Sample a copy of V
                V_prime = self.S(:,1);
                min_loss_prime = V_prime' * (self.eta * L_estimate - Z_prime);
                for i=2:self.N
                    % Find the action that minimizes the perturbed loss
                    if(self.S(:,i)' * (self.eta * L_estimate - Z_prime) < min_loss_prime)
                        V_prime = self.S(:,i);
                        min_loss_prime = V_prime' * (self.eta * L_estimate - Z_prime);
                    end
                end
                r = r .* V_prime;
                if(all(r) == 0)
                    break
                end
            end
            self.K = K;
        end
        
        function self = UpdateLoss(self, l)
            % l is the loss observed from the environement
            l_hat = self.K .* self.V .* l;
            self.L = self.L + l_hat;% L from FPL+GR
            self.L_tilde = self.L_tilde + (1/self.beta)*log(1 + self.beta*l_hat);% L from FPL+GR.P
        end
            
    end
    
end

