pub struct Network<F: Fn(f64) -> f64> {
    activation_fn: F,
    shape: Vec<usize>,
    layers: Vec<Layer>,
}
struct Layer {
    weights: Vec<Vec<f64>>, //[neuron in this layer] [connecting neuron in prev layer]
    biases: Vec<f64>,
}

impl<F: Fn(f64) -> f64> Network<F> {
    pub fn new(shape: Vec<usize>, activation_fn: F) -> Network<F> {
        let mut layers: Vec<Layer> = vec![];
        for i in 1..shape.len() {
            //1..shape.len() bcs the input layer shouldn't be an actual layer
            layers.push(Layer::new(*&shape[i], *&shape[i-1]))
        }
        Network {
            activation_fn,
            shape,
            layers,
        }
    }
    pub fn pulse(&self, input: Vec<f64>) -> Vec<f64> {
        if input.len() < self.shape[0] {panic!("Network was passed more inputs than there are neurons in the first layer of the network");}
        let mut layer_output = input;
        for layer in self.layers.iter() {
            layer_output = layer.pulse(layer_output, self);
        }
        layer_output
    }
}

impl Layer {
    pub fn new(layer_size: usize, prev_layer_size: usize) -> Layer {
        Layer {
            weights: vec![vec![1.0; prev_layer_size];layer_size],
            biases: vec![0.0; layer_size]
        }
    }
    pub fn pulse<F: Fn(f64) -> f64>(&self, input: Vec<f64>, network: &Network<F>) -> Vec<f64> {
        let mut output_buf: Vec<f64> = vec![];
        for neuron_indx in 0..self.biases.len() {
            //For every neuron in layer
            output_buf.push(
                (network.activation_fn)({
                    //Sum the weighted inputs
                    let mut sum: f64 = 0.0;
                    for input_indx in 0..input.len() {
                        //for every input
                        sum += input[input_indx] * *(self.get_weight(neuron_indx, input_indx).unwrap());
                    }
                    sum
                } + self.get_bias(neuron_indx).unwrap())
            );
        }
        output_buf
    }
    pub fn set_weight(&mut self, neuron: usize, prev_layer_neuron: usize, weight: f64) {
        *(self.weights
            .get_mut(neuron)
            .expect("A viable neuron ID")
            .get_mut(prev_layer_neuron)
            .expect("A viable ID to a neuron in the previous layer")) = weight;
    }
    pub fn get_weight(&self, neuron: usize, prev_layer_neuron: usize) -> Option<&f64> {
        Some(self.weights
            .get(neuron)?
            .get(prev_layer_neuron)?)
    }
    pub fn set_bias(&mut self, neuron: usize, bias: f64) {
        *(self.biases.get_mut(neuron).expect("A valid neuron ID")) = bias;
    }
    pub fn get_bias(&self, neuron: usize) -> Option<&f64> {
        Some(self.biases.get(neuron)?)
    }
}