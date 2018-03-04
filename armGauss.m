classdef armGauss < handle
    % % Gaussian arm
    
    properties
        mean; % mean of the gaussian
        var; % variance of the gaussian
    end
    
    methods
        function self = armGauss(mean, var)
            self.mean = mean;
            self.var = var;
        end
        
        function reward = sample(self)
            reward = normrnd(self.mean, self.var);
        end
    end
    
end

