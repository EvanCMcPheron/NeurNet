use std::io::Read;

use neurnet::*;

fn main() {
    let mut buf_rate = String::new();
    std::io::stdin().read_line(&mut buf_rate).unwrap();
    let rate: f64 = buf_rate.trim().parse().unwrap();
    let mut nn = Network::new(
        vec![1, 5, 3, 2, 5, 3, 2, 1],
        |x| if x > 0.0 { x } else { 0.01 * x },
        Some(|x| if x > 0.0 { 1.0 } else { 0.01 }),
        (-2.0, 2.0),
        (-5.0, 5.0),
    );
    let training_inputs = {
        let mut buf: Vec<Vec<f64>> = vec![];
        for i in -1000..=1000 {
            buf.push(vec![(i as f64) / 2.0])
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
    let ds = DataSet::gen_from_fn(
        |x| vec![if x[0] == 0.0 {100.0} else {50.0 / x[0]}],
        training_inputs.clone(),
        testing_inputs.clone(),
    );
    println!("{:?}", nn.test(&ds));
    for i in 0..2000 {
      nn.train(&ds, rate);
      println!("{}: {:?}", i, nn.test(&ds));
    }
    for i in -100..=100 {
        println!("({}, {})", i, nn.pulse(vec![i as f64])[0]);
    }
    let mut buf = String::new();
    println!("Press enter to exit...");
    std::io::stdin().read_line(&mut buf).unwrap();
}
