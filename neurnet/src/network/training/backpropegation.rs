use super::{Layer, Network};
impl<F: Fn(f64) -> f64> Network<F> {
    pub fn test_fn(&self) {
        println!("Functional!");
    }
}
