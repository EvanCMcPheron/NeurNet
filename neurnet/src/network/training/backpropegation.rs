use crate::NetworkFood;

use super::{Layer, Network};
impl Network {
    pub fn get_activation_der(&self, x: f64) -> f64 {
        //! Gets the derivative of the activation function at the given point **x**.
        let h = 0.0001;
        match &self.activation_der {
            Some(derivative) => derivative(x),
            None => {
                ( (self.activation_fn)(x + h/2.0) - (self.activation_fn)(x - h/2.0) ) / h
            }
        }
    }
    pub fn test_point(&self, point: (Vec<f64>, Vec<f64>)) -> Vec<f64> {
        //! Takes a data point (input values, output values) and returns the costs of each individual output neuron for that given point.
        let outputs = self.pulse(point.0);
        let mut cost_vec: Vec<f64> = vec![];
        for i in 0..outputs.len() {
            cost_vec.push(
                {
                    let c = outputs[i] - point.1.get(i).expect("test point has missized output vec");
                    c * c
                }
            );
        }
        cost_vec
    }
    pub fn test_training_set(&self, food: impl NetworkFood) -> Vec<f64> {
        //! Takes a dataset, runs through the engire set of training data and returns the average cost for each neuron.
        let training_data = food.grab_testing_data();
        let mut cost_totals: Vec<f64> = vec![0.0; *self.shape.last().unwrap()];
        for data_pnt_i in 0..training_data.len() {
            let mut i = 0;
            for neuron_value in self.test_point(training_data[data_pnt_i].clone()/* Potential Bottleneck */) {
                cost_totals[i] += neuron_value;
                i += 1;
            }
        }
        for cost in cost_totals.iter_mut() {
            *cost /= training_data.len() as f64;
        }
        cost_totals
    }
}
