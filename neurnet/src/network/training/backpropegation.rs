use crate::NetworkFood;

use super::{Layer, Network};
impl Network {
    pub fn get_activation_der(&self, x: f64) -> f64 {
        //! Gets the derivative of the activation function at the given point **x**.
        let h = 0.0001;
        match &self.activation_der {
            Some(derivative) => derivative(x),
            None => ((self.activation_fn)(x + h / 2.0) - (self.activation_fn)(x - h / 2.0)) / h,
        }
    }
    pub fn test_point(&self, point: &(Vec<f64>, Vec<f64>)) -> Vec<f64> {
        //! Takes a data point (input values, output values) and returns the costs of each individual output neuron for that given point.
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
        //! Takes a dataset, runs through the entire set of training data, and returns the average cost for each neuron.
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
    pub fn train(&mut self, food: &impl NetworkFood, rate: f64) {
        fn average_vec(vec: &Vec<f64>) -> f64 {
            let mut buf = 0.0;
            for i in vec.iter() {
                buf += i;
            }
            buf / vec.len() as f64
        }
        let training_data = food.grab_training_data();
        for data_pnt in training_data.iter() {
            let mut initial_cost = self.test_point(&data_pnt);
            let mut initial_avg = average_vec(&initial_cost);
            for layer in 0..(self.shape.len() - 1) {
                //-1 as to ignore the input layer
                for neuron_in_layer in 0..self.shape[layer + 1] {
                    //Offset by one to make up for the fact that the shape includes the input layer
                    for neuron_in_prev_layer in 0..self.shape[layer] {
                        let current_weight = self
                            .get_weight(layer, neuron_in_layer, neuron_in_prev_layer)
                            .unwrap().clone();
                        self.set_weight(
                            layer,
                            neuron_in_layer,
                            neuron_in_prev_layer,
                            current_weight + rate,
                        );
                        let new_cost = self.test_point(data_pnt);
                        let new_avg = average_vec(&new_cost);
                        let der = new_avg - initial_avg; //Positive means increasing weight is bad
                        self.set_weight(layer, neuron_in_layer, neuron_in_prev_layer, current_weight - der * rate);
                        initial_cost = self.test_point(data_pnt);
                        initial_avg = average_vec(&initial_cost);
                    }
                    let current_bias = self.get_bias(layer, neuron_in_layer).unwrap().clone();
                    self.set_bias(layer, neuron_in_layer, current_bias + rate);
                    let new_cost = self.test_point(data_pnt);
                    let new_avg = average_vec(&new_cost);
                    let der = new_avg - initial_avg; //Positive means increasing weight is bad
                    self.set_bias(layer, neuron_in_layer, current_bias - der * rate);
                    initial_cost = self.test_point(data_pnt);
                    initial_avg = average_vec(&initial_cost);
                }
            }
        }
    }
}
