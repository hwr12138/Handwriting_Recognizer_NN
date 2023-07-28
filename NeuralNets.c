#include "NeuralNets.h"

int train_1layer_net(double sample[INPUTS],int label,double (*sigmoid)(double input), double weights_io[INPUTS][OUTPUTS])
{
 /*
  *  The main training function for 1-layer networks. 
  * 
  *  Inputs:
  *   sample  -  Array with the pixel values for the input digit - in this case a 28x28 image (784 pixels)
  *              with values in [0-255], plus one bias term (last entry in the array) which is always 1
  *   label  -   Correct label for this digit (our target class)
  *   sigmoid -  The sigmoid function being used, which will be either the logistic function or the hyperbolic
  *              tangent. You have to implement the logistic function, but math.h provides tanh() already
  *   weights_io - Array of weights connecting inputs to output neurons, weights[i][j] is the weight from input
  *                i to output neuron j. This array has a size of 785x10.
  *
  *   Return values:
  *     An int in [0,9] corresponding to the class that your current network has chosen for this training
  *   sample.
  */
  // Compute network output for the input sample
  double activations[OUTPUTS];
  feedforward_1layer(sample, sigmoid, weights_io, activations);

  // Compute error for each output neuron
  double errors[OUTPUTS];
  for (int i = 0; i < OUTPUTS; i++) {
    errors[i] = (i == label ? activations[i] - 1 : activations[i]);
  }

  // Update weights using backpropagation
  backprop_1layer(sample, activations, sigmoid, label, weights_io);

  // Return the class that the network has chosen for this training sample
  int predicted_class = classify_1layer(sample, label, sigmoid, weights_io);
  return predicted_class;		// <--- This should return the class for this sample
}

int classify_1layer(double sample[INPUTS],int label,double (*sigmoid)(double input), double weights_io[INPUTS][OUTPUTS])
{
 /*
  *   This function classifies an input sample given the current network weights. It returns a class in
  *  [0,9] corresponding to the digit the network has decided is present in the input sample
  * 
  *  Inputs:
  *   sample  -  Array with the pixel values for the input digit - in this case a 28x28 image (784 pixels)
  *              with values in [0-255], plus one bias term (last entry in the array) which is always 1
  *   label  -   Correct label for this digit (our target class)
  *   sigmoid -  The sigmoid function being used, which will be either the logistic function or the hyperbolic
  *              tangent. You have to implement the logistic function, but math.h provides tanh() already
  *   weights_io - Array of weights connecting inputs to output neurons, weights[i][j] is the weight from input
  *                i to output neuron j. This array has a size of 785x10.
  *
  *   Return values:
  *     An int in [0,9] corresponding to the class that your current network has chosen for this training
  *   sample.
  * 
  */

  double activations[OUTPUTS];
  feedforward_1layer(sample, sigmoid, weights_io, activations);
    
  int predicted_class = 0;
  double max_activation = 0.0;
    
  for (int i = 0; i < OUTPUTS; i++) {
    if (activations[i] > max_activation) {
      max_activation = activations[i];
      predicted_class = i;
    }
  }
    
  return predicted_class;   	// <---	This should return the class for this sample
}

void feedforward_1layer(double sample[785], double (*sigmoid)(double input), double weights_io[INPUTS][OUTPUTS], double activations[OUTPUTS])
{
 /*
  *  This function performs the feedforward pass of the network's computation - it propagates information
  *  from input to output, determines the input to each neuron, and calls the sigmoid function to
  *  calculate neuron activation.
  * 
  *  Inputs:
  *    sample -      The input sample (see above for a description)
  *    sigmoid -     The sigmoid function being used
  *    weights_op -  Array of current network weights
  *    activations - Array where your function will store the resulting activation for each output neuron
  * 
  *  Return values:
  *    Your function must update the 'activations' array with the output value for each neuron
  */ 
  // Calculate the input to each neuron
    for (int j = 0; j < OUTPUTS; j++) {
        double input = 0.0;
        for (int i = 0; i < INPUTS; i++) {
            input += sample[i] * weights_io[i][j];
        }
        // Scale the input and calculate the neuron activation
        double scaled_input = input * SIGMOID_SCALE;
        activations[j] = sigmoid(scaled_input);
    }
}

double sigmoid_deriv(double (*sigmoid)(double input), double x){
  if(sigmoid(0) == 0){
    return 1.0 - pow(sigmoid(x), 2);
  }
  else{
    return sigmoid(x) * (1 - sigmoid(x));
  }
}

void backprop_1layer(double sample[INPUTS], double activations[OUTPUTS], double (*sigmoid)(double input), int label, double weights_io[INPUTS][OUTPUTS])
{
  /*
   *  This function performs the core of the learning process for 1-layer networks. It takes
   *  as input the feed-forward activation for each neuron, the expected label for this training
   *  sample, and the weights array. Then it updates the weights in the array so as to minimize
   *  error across neuron outputs.
   * 
   *  Inputs:
   * 	sample - 	Input sample (see above for details)
   *    activations - 	Neuron outputs as computed above
   *    sigmoid -	Sigmoid function in use
   *    label - 	Correct class for this sample
   *    weights_io -	Network weights
   */
   // Determine the target value for each neuron
    double targets[OUTPUTS];
    for (int i = 0; i < OUTPUTS; i++) {
        if (i == label) {
            targets[i] = 1.0;
        } else {
            targets[i] = 0.0;
        }
    }

    // Compute an error value given the neuron's target
    double errors[OUTPUTS];
    for (int i = 0; i < OUTPUTS; i++) {
        errors[i] = targets[i] - activations[i];
    }

    // Compute the weight adjustment for each weight
    double learning_rate = ALPHA;
    for (int i = 0; i < INPUTS; i++) {
        for (int j = 0; j < OUTPUTS; j++) {
            double delta_weight = learning_rate * errors[j] * sigmoid_deriv(sigmoid, activations[j]) * sample[i];
            weights_io[i][j] += delta_weight;
        }
    }
}

int train_2layer_net(double sample[INPUTS],int label,double (*sigmoid)(double input), int units, double weights_ih[INPUTS][MAX_HIDDEN], 
                      double weights_ho[MAX_HIDDEN][OUTPUTS])
{
  /*
  *  The main training function for 2-layer networks. 
  * 
  *  Inputs:
  *   sample  -  Array with the pixel values for the input digit - in this case a 28x28 image (784 pixels)
  *              with values in [0-255], plus one bias term (last entry in the array) which is always 1
  *   label  -   Correct label for this digit (our target class)
  *   sigmoid -  The sigmoid function being used, which will be either the logistic function or the hyperbolic
  *              tangent. You have to implement the logistic function, but math.h provides tanh() already
  *   units   -  Number of units in the hidden layer
  *   weights_ih - Array of weights connecting inputs to hidden-layer neurons, weights_ih[i][j] is the 
  *                weight from input i to hidden neuron j. This array has a size of units 785 x 10.
  *   weights_ho - Array of weights connecting hidden-layer units to output neurons, weights_ho[i][j] is the 
  *                weight from hidden unit i to output neuron j. This array has a size of units x 10.
  *
  *   Return values:
  *     An int in [0,9] corresponding to the class that your current network has chosen for this training
  *   sample.
  * 
  */

  double activations[OUTPUTS];
  double h_activations[MAX_HIDDEN];
  feedforward_2layer(sample, sigmoid, weights_ih, weights_ho, h_activations, activations, units);

  // Compute error for each output neuron
  double errors[OUTPUTS];
  for (int i = 0; i < OUTPUTS; i++) {
    errors[i] = (i == label ? activations[i] - 1 : activations[i]);
  }

  // Update weights using backpropagation
  backprop_2layer(sample, h_activations, activations, sigmoid, label, weights_ih, weights_ho, units);

  // Return the class that the network has chosen for this training sample
  double max_activation = -INFINITY;
  int predicted_class = -1;
  for (int i = 0; i < OUTPUTS; i++) {
    if (activations[i] > max_activation) {
      max_activation = activations[i];
      predicted_class = i;
    }
  }
  return predicted_class;
}

int classify_2layer(double sample[INPUTS],int label,double (*sigmoid)(double input), int units, double weights_ih[INPUTS][MAX_HIDDEN], 
                    double weights_ho[MAX_HIDDEN][OUTPUTS])
{
 /*
  *   This function takes an input sample and classifies it using the current network weights. It returns
  *  an int in [0,9] corresponding to which digit the network thinks is present in the input sample.
  * 
  *  Inputs:
  *   sample  -  Array with the pixel values for the input digit - in this case a 28x28 image (784 pixels)
  *              with values in [0-255], plus one bias term (last entry in the array) which is always 1
  *   label   -  Correct label for this digit (our target class)
  *   sigmoid -  The sigmoid function being used, which will be either the logistic function or the hyperbolic
  *              tangent. You have to implement the logistic function, but math.h provides tanh() already
  *   units   -  Number of units in the hidden layer
  *   weights_ih - Array of weights connecting inputs to hidden-layer neurons, weights_ih[i][j] is the 
  *                weight from input i to hidden neuron j. This array has a size of units 785 x 10.
  *   weights_ho - Array of weights connecting hidden-layer units to output neurons, weights_ho[i][j] is the 
  *                weight from hidden unit i to output neuron j. This array has a size of units x 10.
  *
  *   Return values:
  *     An int in [0,9] corresponding to the class that your current network has chosen for this training
  *   sample.
  * 
  */
  double activations[OUTPUTS];
  double h_activations[MAX_HIDDEN];
  feedforward_2layer(sample, sigmoid, weights_ih, weights_ho, h_activations, activations, units);
  int predicted_class = 0;
  double max_activation = 0.0;
  for (int i = 0; i < OUTPUTS; i++) {
    if (activations[i] > max_activation) {
      max_activation = activations[i];
      predicted_class = i;
    }
  }
  return predicted_class; 
}


void feedforward_2layer(double sample[INPUTS], double (*sigmoid)(double input), double weights_ih[INPUTS][MAX_HIDDEN], 
                        double weights_ho[MAX_HIDDEN][OUTPUTS], double h_activations[MAX_HIDDEN],double activations[OUTPUTS], int units)
{
 /*
  *  The feedforward part of the two-layer network's computation.
  * 
  *  Inputs:
  *    sample -      The input sample (see above for a description)
  *    sigmoid -     The sigmoid function being used
  *    weights_ih -  Array of current input-to-hidden weights
  *    weights_ho -  Array of current hidden-to-output weights
  *    h_activations - Array of hidden layer unit activations
  *    activations   - Array of activations for output neurons
  *    units -         Number of units in the hidden layer
  */ 
  for (int hid_n = 0; hid_n < units; hid_n++) {
    double acti = 0;
    for (int in = 0; in < INPUTS - 1; in++) {
      acti += sample[in] * weights_ih[in][hid_n];
    }
    h_activations[hid_n] = sigmoid(SIGMOID_SCALE * acti);
  }
  for (int out_n = 0; out_n < OUTPUTS; out_n++) {
    double acti = 0;
    for (int in = 0; in < units; in++) {
      acti += h_activations[in] * weights_ho[in][out_n];
    }
    activations[out_n] = sigmoid(SIGMOID_SCALE * (MAX_HIDDEN/units) * acti);
  }
}

void backprop_2layer(double sample[INPUTS],double h_activations[MAX_HIDDEN], double activations[OUTPUTS], double (*sigmoid)(double input), 
                      int label, double weights_ih[INPUTS][MAX_HIDDEN], double weights_ho[MAX_HIDDEN][OUTPUTS], int units)
{
  /*
   *  This function performs the core of the learning process for 2-layer networks. It performs
   *  the weights update as discussed in lecture. 
   * 
   *  Inputs:
   * 	  sample - 	Input sample (see above for details)
   *    h_activations - Hidden-layer activations
   *    activations -   Output-layer activations
   *    sigmoid -	Sigmoid function in use
   *    label - 	Correct class for this sample
   *    weights_ih -	Network weights from inputs to hidden layer
   *    weights_ho -    Network weights from hidden layer to output layer
   *    units -         Number of units in the hidden layer
   */
  double cache[MAX_HIDDEN][OUTPUTS];
  for (int i = 0; i < MAX_HIDDEN; i ++) {
    for (int j = 0; j < OUTPUTS; j++) {
      cache[i][j] = weights_ho[i][j];
    }
  }
  for (int out_n = 0; out_n < OUTPUTS; out_n++) {
    for (int in = 0; in < units; in++) {
      double e_to_o = 0 - activations[out_n];
      if (out_n == label) {
        e_to_o = 1 - activations[out_n];
      }
      double o_to_a = sigmoid_deriv(sigmoid, h_activations[out_n]);
      weights_ho[in][out_n] += ALPHA * (h_activations[in] * o_to_a * e_to_o);
    }
  }
  for (int hid_n = 0; hid_n < units; hid_n++) {
    for (int in = 0; in < INPUTS - 1; in++) {
      double e_to_o = 0.0;
      for (int j = 0; j < OUTPUTS; j++) {
        double t = 0.0 - activations[j];
        if (j == label) {
          t = 1.0 - activations[j];
        }
        e_to_o += cache[hid_n][j] * sigmoid_deriv(sigmoid, activations[j]) * t;
      }
      double o_to_a = sigmoid_deriv(sigmoid, h_activations[hid_n]);
      weights_ih[in][hid_n] += ALPHA * (sample[in] * o_to_a * e_to_o);
    }
  }
}

double logistic(double input)
{
  // This function returns the value of the logistic function evaluated on input
  return 1.0 / (1.0 + exp(-input));
}
