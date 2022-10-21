use super::files::{parse_neur_file, write_neur_file};
pub mod training;
pub struct Network {
    activation_fn: Box<dyn Fn(f64) -> f64>,
    shape: Vec<usize>,
    layers: Vec<Layer>,
}
pub struct Layer {
    weights: Vec<Vec<f64>>, //[neuron in this layer] [connecting neuron in prev layer]
    biases: Vec<f64>,
}

impl Network {
    pub fn new<AF: Fn(f64) -> f64 + 'static>(
        shape: Vec<usize>,
        activation_fn: AF,
        weights_range: (f64, f64),
        biases_range: (f64, f64),
    ) -> Network {
        let mut layers: Vec<Layer> = vec![];
        for i in 1..shape.len() {
            //1..shape.len() bcs the input layer shouldn't be an actual layer
            layers.push({
                let mut layer = Layer::new(*&shape[i], *&shape[i - 1]);
                layer.randomize(weights_range, biases_range);
                layer
            })
        }
        Network {
            activation_fn: Box::new(activation_fn),
            shape,
            layers,
        }
    }
    pub fn save_safe(&self, name: &str) {
      //! ```
      //! let network1 = Network::new(
      //!   vec![1, 2, 1],
      //!   |x| if x > 0.0 { x } else { 0.01 * x },
      //!   (-2.0, 2.0),
      //!   (-5.0, 5.0),
      //! );
      //! network1.save_safe(&"network");
      //! let network2 = Network::load(
      //!   &"network.neur",
      //!   |x| if x > 0.0 { x } else { 0.01 * x },
      //! ).unwrap();
      //! 
      //! ```
        match self.save(&{ let mut string = String::from(name); string.push_str(".neur"); string }) {
            Some(_) => (),
            None => {
                let mut i: usize = 0;
                loop {
                    match self
                        .save(&{ let mut string = String::from(name); string.push_str(&i.to_string()); string.push_str(".neur"); string })
                    {
                        Some(_) => break,
                        None => (),
                    }
                    i += 1;
                }
            }
        }
    }
    pub fn save(&self, path: &str) -> Option<()> {
        let mut data: (Vec<usize>, Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) =
            (self.get_shape().clone(), vec![], vec![]);
        let mut layer_i = 0;
        for layer in self.layers.iter() {
            data.1.push(vec![]);
            data.2.push(vec![]);
            for neuron_i in 0..layer.len() {
                data.1[layer_i].push(vec![]);
                for prev_neuron_i in 0..layer.prev_layer_len() {
                    data.1[layer_i][neuron_i]
                        .push(*self.get_weight(layer_i, neuron_i, prev_neuron_i).unwrap());
                }
                data.2[layer_i].push(*self.get_bias(layer_i, neuron_i).unwrap());
            }
            layer_i += 1;
        }
        write_neur_file(path, data)
    }
    pub fn load<AF: Fn(f64) -> f64 + 'static>(
        path: &str,
        activation_fn: AF,
    ) -> Option<Network> {
        let data = parse_neur_file(path)?;
        let mut network = Network::new(
            data.0.clone(),
            activation_fn,
            (0.0, 0.0),
            (0.0, 0.0),
        );
        let mut layer_i = 0;
        for layer in network.get_layers_mut().iter_mut() {
            for neuron_i in 0..layer.len() {
                for prev_neuron_i in 0..layer.prev_layer_len() {
                    layer.set_weight(
                        neuron_i,
                        prev_neuron_i,
                        data.1[layer_i][neuron_i][prev_neuron_i],
                    );
                }
                layer.set_bias(neuron_i, data.2[layer_i][neuron_i]);
            }
            layer_i += 1;
        }
        Some(network)
    }
    pub fn pulse(&self, input: Vec<f64>) -> Vec<f64> {
        if input.len() < self.shape[0] {
            panic!("Network was passed more inputs than there are neurons in the first layer of the network");
        }
        let mut layer_output = input;
        for layer in self.layers.iter() {
            layer_output = layer.pulse(layer_output, &self.activation_fn);
        }
        layer_output
    }
    pub fn set_weight(
        &mut self,
        layer: usize,
        neuron: usize,
        prev_layer_neuron: usize,
        weight: f64,
    ) {
        //! Sets the weight of a neuron connection between a neuron in *layer* and a neuron in the previous layer. Note that the input layer isn't counted as a layer, so layer=0 would actually be accessing the second layer.
        //! # Panics
        //! <ul>
        //! <li> Attempting to mutate the weight to or from a non-existant neuron.
        //! </ul>
        self.layers
            .get_mut(layer)
            .unwrap()
            .set_weight(neuron, prev_layer_neuron, weight);
    }
    pub fn get_weight(
        &self,
        layer: usize,
        neuron: usize,
        prev_layer_neuron: usize,
    ) -> Option<&f64> {
        self.layers
            .get(layer)?
            .get_weight(neuron, prev_layer_neuron)
    }
    pub fn set_bias(&mut self, layer: usize, neuron: usize, bias: f64) {
        self.layers.get_mut(layer).unwrap().set_bias(neuron, bias);
    }
    pub fn get_bias(&self, layer: usize, neuron: usize) -> Option<&f64> {
        self.layers.get(layer)?.get_bias(neuron)
    }
    pub fn get_shape(&self) -> &Vec<usize> {
        &self.shape
    }
    pub fn get_layers(&self) -> &Vec<Layer> {
        &self.layers
    }
    pub fn get_layers_mut(&mut self) -> &mut Vec<Layer> {
        &mut self.layers
    }
    pub fn randomize(&mut self, weights_range: (f64, f64), biases_range: (f64, f64)) {
        for layer in self.layers.iter_mut() {
            layer.randomize(weights_range, biases_range);
        }
    }
}

impl Layer {
    pub fn new(layer_size: usize, prev_layer_size: usize) -> Layer {
        Layer {
            weights: vec![vec![1.0; prev_layer_size]; layer_size],
            biases: vec![0.0; layer_size],
        }
    }
    pub fn pulse(&self, input: Vec<f64>, activation_fn: &Box<dyn Fn(f64) -> f64>) -> Vec<f64> {
        let mut output_buf: Vec<f64> = vec![];
        for neuron_indx in 0..self.biases.len() {
            //For every neuron in layer
            output_buf.push((activation_fn)(
                {
                    //Sum the weighted inputs
                    let mut sum: f64 = 0.0;
                    for input_indx in 0..input.len() {
                        //for every input
                        sum += input[input_indx]
                            * *(self.get_weight(neuron_indx, input_indx).unwrap());
                    }
                    sum
                } + self.get_bias(neuron_indx).unwrap(),
            ));
        }
        output_buf
    }
    pub fn set_weight(&mut self, neuron: usize, prev_layer_neuron: usize, weight: f64) {
        *(self
            .weights
            .get_mut(neuron)
            .expect("A viable neuron ID")
            .get_mut(prev_layer_neuron)
            .expect("A viable ID to a neuron in the previous layer")) = weight;
    }
    pub fn get_weight(&self, neuron: usize, prev_layer_neuron: usize) -> Option<&f64> {
        Some(self.weights.get(neuron)?.get(prev_layer_neuron)?)
    }
    pub fn set_bias(&mut self, neuron: usize, bias: f64) {
        *(self.biases.get_mut(neuron).expect("A valid neuron ID")) = bias;
    }
    pub fn get_bias(&self, neuron: usize) -> Option<&f64> {
        Some(self.biases.get(neuron)?)
    }
    pub fn randomize(&mut self, weights_range: (f64, f64), biases_range: (f64, f64)) {
        fn rand_float(range: (f64, f64)) -> f64 {
            use rand::random;
            let mut buf: f64 = random();
            buf *= range.1 - range.0;
            buf += range.0;
            buf
        }
        for neuron in self.weights.iter_mut() {
            for weight in neuron.iter_mut() {
                *weight = rand_float(weights_range);
            }
        }
        for bias in self.biases.iter_mut() {
            *bias = rand_float(biases_range);
        }
    }
    pub fn len(&self) -> usize {
        self.biases.len()
    }
    pub fn prev_layer_len(&self) -> usize {
        self.weights.get(0).unwrap().len()
    }
}
