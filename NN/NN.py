import numpy as np

class Neural_Network(object):
  
    def __init__(self, n_layer_nodes, n_hidden, weight_interval, weight_interval_output):
        # n_layer_nodes = [1,0,0,0,1]
        
        self.n_hidden = n_hidden

        # activation-functions
        self.sigmoid = lambda x : (1/(1 + np.exp(-x))) 

        self.tanh = lambda x : np.tanh(x)
        
        self.relu = lambda x : np.maximum(x, 0)

        # iniialize layers with random weights          
        weights_input_to_hidden = np.random.uniform(weight_interval[0], weight_interval[1], 
                                                   (n_layer_nodes[0], n_layer_nodes[1]))                                                   

        bias_output = np.random.uniform(low = weight_interval[0], high = weight_interval[1], size = n_layer_nodes[4])
        bias_hidden_1 = np.random.uniform(low = weight_interval[0], high = weight_interval[1],size = n_layer_nodes[1])    

        if self.n_hidden == 1:
            weights_hidden_to_output = np.random.uniform(weight_interval_output[0], weight_interval_output[1], 
                                                        (n_layer_nodes[1], n_layer_nodes[4]))         
            self.weights = [weights_input_to_hidden, weights_hidden_to_output]     
            self.bias = [bias_hidden_1, bias_output]    
            return

        weights_hidden_1_to_hidden_2 = np.random.uniform(weight_interval[0], weight_interval[1], 
                                                    (n_layer_nodes[1], n_layer_nodes[2]))  
        bias_hidden_2 = np.random.uniform(low = weight_interval[0], high = weight_interval[1], size = n_layer_nodes[2]) 

        if self.n_hidden == 2:
            weights_hidden_to_output = np.random.uniform(weight_interval_output[0], weight_interval_output[1], 
                                               (n_layer_nodes[2], n_layer_nodes[4]))         
            self.weights = [weights_input_to_hidden, weights_hidden_1_to_hidden_2, weights_hidden_to_output]    
            self.bias = [bias_hidden_1, bias_hidden_2, bias_output]
            return
                               
        weights_hidden2_to_hidden3 = np.random.uniform(weight_interval[0], weight_interval[1], 
                                                    (n_layer_nodes[2], n_layer_nodes[3])) 
        bias_hidden_3 = np.random.uniform(low = weight_interval[0], high = weight_interval[1],size = n_layer_nodes[3]) 

        if self.n_hidden == 3:
            weights_hidden_to_output = np.random.uniform(weight_interval_output[0], weight_interval_output[1], 
                                                        (n_layer_nodes[3], n_layer_nodes[4]))  
            self.weights = [weights_input_to_hidden, weights_hidden_1_to_hidden_2, weights_hidden2_to_hidden3, weights_hidden_to_output]
            self.bias = [bias_hidden_1, bias_hidden_2, bias_hidden_3, bias_output]      
            return 
                
    def run_forward_pass(self, observation):
        if self.n_hidden == 1:
            hidden_inputs_layer_1 = np.dot(observation , self.weights[0] ) + self.bias[0]  # signals into hidden layer
            hidden_outputs_layer_1 = self.relu(hidden_inputs_layer_1) # signals from hidden layer 1
            
            final_in = np.dot(hidden_outputs_layer_1 , self.weights[1]) + self.bias[1]  # signals into output layer
            final_out = self.tanh(final_in) # signals from output layer -> actions

        elif self.n_hidden == 2:
            hidden_inputs_layer_1 = np.dot(observation , self.weights[0] ) + self.bias[0]
            hidden_outputs_layer_1 = self.relu(hidden_inputs_layer_1)

            hidden_inputs_layer_2 = np.dot(hidden_outputs_layer_1, self.weights[1]) + self.bias[1] 
            hidden_2_out = self.relu(hidden_inputs_layer_2)
            
            final_in = np.dot(hidden_2_out , self.weights[2]) + self.bias[2]
            final_out = self.tanh(final_in)
        
        elif self.n_hidden == 3:
            hidden_1_in = np.dot(observation , self.weights[0] ) + self.bias[0] 
            hidden_1_out = self.relu(hidden_1_in) 

            hidden_2_in = np.dot(hidden_1_out, self.weights[1]) + self.bias[1] 
            hidden_2_out = self.relu(hidden_2_in)

            hidden_3_in = np.dot(hidden_2_out, self.weights[2]) + self.bias[2] 
            hidden_3_out = self.relu(hidden_3_in)

            final_in = np.dot(hidden_3_out , self.weights[3]) + self.bias[3]
            final_out = self.tanh(final_in)            
            
        return final_out
    
    def set_weights(self, weights):
        self.weights = weights
        return 0