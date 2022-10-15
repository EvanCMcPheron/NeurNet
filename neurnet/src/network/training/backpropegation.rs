use super::{Layer, Network};
impl Network {
    pub fn get_activation_der(&self, x: f64) -> f64 {
        let h = 0.0001;
        match &self.activation_der {
            Some(derivative) => derivative(x),
            None => {
                ( (self.activation_fn)(x + h/2.0) - (self.activation_fn)(x - h/2.0) ) / h
            }
        }
    }
}
