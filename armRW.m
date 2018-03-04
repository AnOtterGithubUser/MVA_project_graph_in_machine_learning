classdef armRW < handle
    % % Random walk arm
    
    properties
        l0 % starting point
        l % current point
        sigma % variance of the random walk
        walk % trajectory of the random walk
    end
    
    methods
        function self = armRW(l0, sigma)
            self.l0 = l0;
            self.sigma = sigma;
            self.l = l0;
            self.walk = [self.l];
        end
        
        function reward = sample(self)
            self.l = self.l + normrnd(0, self.sigma); % Random gaussian walk
            reward = self.l;
            self.walk = [self.walk, self.l];
        end
        
    end
    
end

