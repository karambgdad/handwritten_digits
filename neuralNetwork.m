classdef neuralNetwork
    % neuralNetwork: Class for a three layer nN
    % A detailed explanation goes here ;-)

    properties
        inodes; % number of input nodes
        hnodes; % number of hidden nodes
        onodes; % number of output nodes
        lr;     % learning rate
        w_ih;   % link weights between input layer and hidden layer
        w_ho;   % link weights between hidden layer and output layer
    end

    methods
        function self = neuralNetwork(inputnodes, hiddennodes, outputnodes, learningrate)
            % neuralNetwork: Construct an instance of this class
            % Your code goes here. 
        end
            
        % Query the neural network. The function takes a vector of inputs
        % and computes the outputs, i.e. a "prediction".
        function final_outputs = query(self, inputs)
            % your code goes here.
        end

        % Train the neural network. This function learns the link weights
        % from the training data.
        function self = train(self, inputs, targets)
            % your code goes here
        end
    
    end

    methods(Static)
        function out = actfun(x)
            % The code for the activation function goes here.
        end
    end

end
