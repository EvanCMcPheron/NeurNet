use neurnet::*;

fn main() {
    let mut nn = Network::new(
        vec![1, 2, 1],
        |x| if x > 0.0 {x} else {0.01 * x},
        Some(|x| if x > 0.0 {1.0} else {0.01}),
        (-2.0, 2.0),
        (-5.0, 5.0),
    );
    let training_inputs = {
        let mut buf: Vec<Vec<f64>> = vec![];
        for i in -100..=100 {
            buf.push(vec![i as f64])
        }
        buf
    };
    let testing_inputs = {
        let mut buf: Vec<Vec<f64>> = vec![];
        for i in -100..=100 {
            buf.push(vec![(i as f64) + 0.5])
        }
        buf
    };
    let ds = DataSet::gen_from_fn(|x| vec![0.5 * x[0]], training_inputs, testing_inputs);
    println!("{:?}", nn.test_training_set(ds));
}
