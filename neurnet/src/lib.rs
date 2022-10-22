//! # NeurNet
//! A crate used for creating neural networks and training them. Currently only supports trainging through gradient descent.
//! # Example
//! ```
//! use neurnet::*;
//! let mut nn = Network::new(
//!     vec![1, 5, 4, 5, 4, 6, 2],
//!     |x| if x > 0.0 { x } else { 0.01 * x },
//!     (-2.0, 2.0),
//!     (-5.0, 5.0),
//! );
//! let training_inputs: Vec<Vec<f64>> = (-500..500).map(|x| vec![(x as f64) / 5.0]).collect();
//! let testing_inputs: Vec<Vec<f64>> = (-100..100).map(|x| vec![x as f64]).collect();
//! let ds = DataSet::gen_from_fn(
//!     |x| vec![0.5 * x[0], 2.0 * x[0]],
//!     training_inputs,
//!     testing_inputs,
//! );
//! nn.train_loop(&ds, 0.001, 0.3, 500, Some(10));
//! for i in -100..=100 {
//!     println!("({}, {:?})", i, nn.pulse(vec![i as f64]));
//! }
//! let mut buf = String::new();
//! nn.save_safe("test");
//! ```

pub mod network;
pub use network::training::data::{DataSet, NetworkFood};
pub use network::Network;
mod files;
