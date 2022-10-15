use neurnet::*;

fn main() {
    let mut nn = Network::new(
        vec![2, 2, 1],
        Box::new(|x| if x > 0.0 {x} else {0.01 * x}),
        Some(Box::new(|x| if x > 0.0 {1.0} else {0.01})),
        (-2.0, 2.0),
        (-5.0, 5.0),
    );
    let input = vec![1.0, 1.0];
    println!(
        "Network Input: {:?}\nNetwork Output: {:?}",
        &input,
        nn.pulse(input.clone())
    );
    let ranges: ((f64, f64), (f64, f64)) = ((-2.0, 2.0), (-5.0, 5.0));
    for _ in 0..1000 {
        nn.randomize(ranges.0, ranges.1);
        println!(
            "\nNetwork Input: {:?}\nNetwork Output: {:?}",
            &input,
            nn.pulse(input.clone())
        );
        let check: (f64, f64) = (
            *nn.get_weight(0, 0, 0).unwrap(),
            *nn.get_bias(0, 0).unwrap(),
        );
        println!("Some weight: {}\nSome Bias: {}", check.0, check.1,);
        if check.0 < ranges.0 .0
            || check.0 > ranges.0 .1
            || check.1 < ranges.1 .0
            || check.1 > ranges.1 .1
        {
            panic!("something is wrong!");
        }
    }
    println!("Activation Derivative: {}", nn.get_activation_der(1.0));
}
