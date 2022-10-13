use neurnet::*;

fn main() {
    let mut nn = network::Network::new(
        vec![2,2,1], 
        |x: f64| -> f64 {1.0 / (1.0 + (-x).exp())}
    );
    let input = vec![1.0, 1.0];
    println!("Network Input: {:?}\nNetwork Output: {:?}", &input, nn.pulse(input.clone()));
}