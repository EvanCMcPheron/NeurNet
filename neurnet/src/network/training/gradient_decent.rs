use crate::NetworkFood;

use super::Network;
impl Network {
    fn get_activation_der(&self, x: f64) -> f64 {
        let h = 0.0001;
        ((self.activation_fn)(x + h / 2.0) - (self.activation_fn)(x - h / 2.0)) / h
    }
    pub fn test_point(&self, point: &(Vec<f64>, Vec<f64>)) -> Vec<f64> {
        //! Takes a data point (input values, output values) and returns the costs of each output neuron for that given point.
        let point = point.clone(); /* Potential Bottleneck */
        let outputs = self.pulse(point.0);
        let mut cost_vec: Vec<f64> = vec![];
        for i in 0..outputs.len() {
            cost_vec.push({
                let c = outputs[i] - point.1.get(i).expect("test point has missized output vec");
                c * c
            });
        }
        cost_vec
    }
    pub fn test(&self, food: &impl NetworkFood) -> Vec<f64> {
        //! Takes a dataset, runs through the entire set of testing data, and returns the average cost for each neuron.
        let testing_data = food.grab_testing_data();
        let mut cost_totals: Vec<f64> = vec![0.0; *self.shape.last().unwrap()];
        for data_pnt_i in 0..testing_data.len() {
            let mut i = 0;
            for neuron_value in self.test_point(&testing_data[data_pnt_i]) {
                cost_totals[i] += neuron_value;
                i += 1;
            }
        }
        for cost in cost_totals.iter_mut() {
            *cost /= testing_data.len() as f64;
        }
        cost_totals
    }
    pub fn train_loop(
        &mut self,
        food: &impl NetworkFood,
        rate: f64,
        used_data_fraction: f64,
        iterations: usize,
        iteration_per_cost_print: Option<usize>,
    ) {
        //! The method to use for training a network using gradient descent. 
        //! <ul>
        //! <li> Food is the dataset that will be used to train the network. 
        //! <li> The rate is how big of changes the network will make to lower its costs. If it is too low the network will learn slowly, and if it is too high the network will just jump around randomly without making any notable progress. The rate is recommended to be around 0.0005 to 0.001. 
        //! <li> The used data fraction is the amount of data to be used each iteration (so 0.3 will mean 30% of the data will be used every iteration). A higher value would make the network more consistantly lower its cost, and a lower value would make the network learn faster and avoid potential settle-points.
        //! <li> Iterations is the amount of times to run through the dataset to train the network.
        //! <li> Iterations per cost print is the amount of iterations for each print to the console. If it is None, then nothing will be printed. If it is Some(10), then the cost vector will be printed every 10 generations.
        //! </ul>
        if let Some(_) = iteration_per_cost_print {
            println!("Initial Cost Vector: {:?}", self.test(food));
        }
        for i in 0..iterations {
            self.train(food, rate, used_data_fraction);
            if let Some(gens) = iteration_per_cost_print {
                if i % gens == 0 {
                    let test_results = self.test(food);
                    if test_results[0] == f64::NAN {
                        self.randomize((-2.0, 2.0), (-5.0, 5.0))
                    }
                    println!("Generation {i}: {:?}", test_results);
                }
            }
        }
    }
    fn train(&mut self, food: &impl NetworkFood, rate: f64, used_data_fraction: f64) {
        let training_data = food.grab_training_data();
        for data_pnt in training_data.iter() {
            let random: f64 = rand::random();
            if random < used_data_fraction {
                let mut initial_cost = self.test_point(&data_pnt);
                if initial_cost[0] == f64::NAN {
                    self.randomize((-2.0, 2.0), (-5.0, 5.0));
                } //You only have to check the first output b/c if one output is nan then all will be.
                for output_i in 0..(*self.shape.last().unwrap()) {
                    for layer in 0..(self.shape.len() - 1) {
                        //-1 as to ignore the input layer
                        for neuron_in_layer in 0..self.shape[layer + 1] {
                            //Offset by one to make up for the fact that the shape includes the input layer
                            for neuron_in_prev_layer in 0..self.shape[layer] {
                                let current_weight = self
                                    .get_weight(layer, neuron_in_layer, neuron_in_prev_layer)
                                    .unwrap()
                                    .clone();
                                self.set_weight(
                                    layer,
                                    neuron_in_layer,
                                    neuron_in_prev_layer,
                                    current_weight + rate,
                                );
                                let new_cost = self.test_point(data_pnt); //Can potentially be eliminated by backwards propegation
                                let der = new_cost[output_i] - initial_cost[output_i]; //Positive means increasing weight is bad
                                self.set_weight(
                                    layer,
                                    neuron_in_layer,
                                    neuron_in_prev_layer,
                                    current_weight - der * rate,
                                );
                                initial_cost = self.test_point(data_pnt);
                            }
                            let current_bias =
                                self.get_bias(layer, neuron_in_layer).unwrap().clone();
                            self.set_bias(layer, neuron_in_layer, current_bias + rate);
                            let new_cost = self.test_point(data_pnt);
                            let der = new_cost[output_i] - initial_cost[output_i]; //Positive means increasing weight is bad
                            self.set_bias(layer, neuron_in_layer, current_bias - der * rate);
                            initial_cost = self.test_point(data_pnt);
                        }
                    }
                }
            }
        }
    }
}
