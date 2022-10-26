use std::fs::{File, OpenOptions};
use std::io::{Read, Write};

fn read_file(path: &str) -> Option<String> {
    let mut input = match File::open(path) {
        Ok(input) => input,
        Err(_) => {
            println!("Invalid File Path");
            return None;
        }
    };
    let mut buf = String::new();
    if let Err(_) = input.read_to_string(&mut buf) {
        println!("Could not read file");
        return None;
    }
    Some(buf)
}

fn read_file_bytes(path: &str) -> Option<Vec<u8>> {
    let mut input = match File::open(path) {
        Ok(input) => input,
        Err(_) => {
            println!("Invalid File Path");
            return None;
        }
    };
    let mut buf = vec![];
    if let Err(_) = input.read_to_end(&mut buf) {
        println!("Could not read file");
        return None;
    }
    
    Some(buf)
}

fn write_file(path: &str, contents: &str) -> Option<()> {
    let mut file = match OpenOptions::new().write(true).open(path) {
        Ok(file) => file,
        Err(_) => match File::create(&path) {
            Ok(file) => file,
            Err(_) => return None,
        },
    };
    let data: Box<[u8]> = contents.as_bytes().iter().map(|x| *x).collect();
    match file.write_all(&data) {
        Ok(_) => Some(()),
        Err(_) => None,
    }
}

fn write_file_bytes(path: &str, contents: Vec<u8>) -> Option<()> {
    let mut file = match OpenOptions::new().write(true).open(path) {
        Ok(file) => file,
        Err(_) => match File::create(&path) {
            Ok(file) => file,
            Err(_) => return None,
        },
    };
    let data: Box<[u8]> = contents.iter().map(|x| *x).collect();
    match file.write_all(&data) {
        Ok(_) => Some(()),
        Err(_) => None,
    }
}

pub fn write_dset_file(
    path: &str,
    training_count: u32,
    testing_count: u32,
    input_size: u32,
    output_size: u32,
    training_points: Vec<(Vec<f64>, Vec<f64>)>,
    testing_points: Vec<(Vec<f64>, Vec<f64>)>,
) -> Option<()> {
    let mut file_buf: Vec<u8> = vec![];
    file_buf.extend_from_slice(&training_count.to_be_bytes());
    file_buf.extend_from_slice(&testing_count.to_be_bytes());
    file_buf.extend_from_slice(&input_size.to_be_bytes());
    file_buf.extend_from_slice(&output_size.to_be_bytes());

    for data_point in training_points {
        for input in data_point.0 {
            file_buf.extend_from_slice(&input.to_be_bytes());
        }
        for output in data_point.1 {
            file_buf.extend_from_slice(&output.to_be_bytes());
        }
    }
    for data_point in testing_points {
        for input in data_point.0 {
            file_buf.extend_from_slice(&input.to_be_bytes());
        }
        for output in data_point.1 {
            file_buf.extend_from_slice(&output.to_be_bytes());
        }
    }

    write_file_bytes(path, file_buf)?;
    Some(())
}

pub fn read_dset_file(path: &str) -> Option<(Vec<(Vec<f64>, Vec<f64>)>, Vec<(Vec<f64>, Vec<f64>)>)> {
    fn grab_u32(data: &mut Vec<u8>) -> u32 {
        let mut buf: [u8; 4] = [0; 4];
        for i in 0..4 {
            buf[i] = data.remove(0);
        }
        u32::from_be_bytes(buf)
    }
    fn grab_f64(data: &mut Vec<u8>) -> f64 {
        let mut buf: [u8; 8] = [0; 8];
        for i in 0..8 {
            buf[i] = data.remove(0);
        }
        f64::from_be_bytes(buf)
    }
    
    let mut data = read_file_bytes(path)?;

    let training_count = grab_u32(&mut data);
    let testing_count = grab_u32(&mut data);
    let input_size = grab_u32(&mut data);
    let output_size = grab_u32(&mut data);

    let mut buf = (vec![], vec![]);

    for training_pnt in 0..training_count {
        buf.0.push((vec![], vec![]));
        for input in 0..input_size {
            buf.0[training_pnt as usize].0.push(grab_f64(&mut data));
        }
        for output in 0..output_size {
            buf.0[training_pnt as usize].1.push(grab_f64(&mut data));
        }
    }
    for testing_pnt in 0..testing_count {
        buf.1.push((vec![], vec![]));
        for input in 0..input_size {
            buf.1[testing_pnt as usize].0.push(grab_f64(&mut data));
        }
        for output in 0..output_size {
            buf.1[testing_pnt as usize].1.push(grab_f64(&mut data));
        }
    }

    Some(buf)
}

pub fn parse_neur_file(path: &str) -> Option<(Vec<usize>, Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>)> {
    let data: String = read_file(path)?;
    fn rm_whitespace(x: String) -> String {
        let chars = x.chars();
        let mut buf = String::new();
        for char in chars {
            if char != ' ' && char != '\n' {
                buf.push(char);
            }
        }
        buf
    }
    fn trim_trailing_comma(x: String) -> String {
        let mut char_vec: Vec<char> = x.chars().collect();
        if char_vec.last().unwrap() == &',' {
            char_vec.remove(char_vec.len() - 1);
        }
        char_vec.iter().collect()
    }
    fn vectorize_lists(x: String) -> Vec<String> {
        let chars = x.chars();
        let mut buf: Vec<String> = vec![];
        let mut char_buf: Vec<char> = vec![];
        let mut depth: i32 = 0;
        for char in chars {
            if char == ']' {
                depth -= 1;
                if depth == 0 {
                    buf.push(char_buf.iter().collect::<String>());
                } else {
                    char_buf.push(char);
                }
            } else if char == '[' {
                depth += 1;
                if depth == 1 {
                    char_buf = vec![];
                } else {
                    char_buf.push(char);
                }
            } else {
                char_buf.push(char);
            }
        }
        buf
    }
    fn vectorize_numlist(x: String) -> Vec<String> {
        let chars = x.chars();
        let mut char_buf: Vec<char> = vec![];
        let mut buf: Vec<String> = vec![];
        for char in chars {
            if char == ',' {
                buf.push(char_buf.iter().collect::<String>());
                char_buf = vec![];
            } else {
                char_buf.push(char);
            }
        }
        if !char_buf.is_empty() {
            buf.push(char_buf.iter().collect::<String>());
        }
        buf
    }
    fn convert_f64_list(x: Vec<String>) -> Vec<f64> {
        let mut buf: Vec<f64> = vec![];
        for string in x.iter() {
            buf.push(
                string
                    .parse::<f64>()
                    .expect("Failed to parse string into f64"),
            );
        }
        buf
    }
    fn convert_usize_list(x: Vec<String>) -> Vec<usize> {
        let mut buf: Vec<usize> = vec![];
        for string in x.iter() {
            buf.push(
                string
                    .parse::<usize>()
                    .expect("Failed to parse string into f64"),
            );
        }
        buf
    }
    let vectorized_input = vectorize_lists(trim_trailing_comma(rm_whitespace(data)));
    let shape: Vec<usize> = convert_usize_list(vectorize_numlist(vectorized_input[0].clone()));
    let weights: Vec<Vec<Vec<f64>>> = {
        let layer_strings = vectorize_lists(vectorized_input[1].clone());
        let mut network_buf: Vec<Vec<Vec<f64>>> = vec![];
        for layer_string in layer_strings {
            let mut layer_buf: Vec<Vec<f64>> = vec![];
            let neuron_strings = vectorize_lists(layer_string);
            for neuron_string in neuron_strings {
                let connection_strings = vectorize_numlist(neuron_string);
                layer_buf.push(convert_f64_list(connection_strings));
            }
            network_buf.push(layer_buf);
        }
        network_buf
    };
    let biases = {
        let layer_strings = vectorize_lists(vectorized_input[2].clone());
        let mut buf: Vec<Vec<f64>> = vec![];
        for layer_string in layer_strings {
            let neuron_strings = vectorize_numlist(layer_string);
            buf.push(convert_f64_list(neuron_strings));
        }
        buf
    };
    Some((shape, weights, biases))
}

pub fn write_neur_file(
    path: &str,
    data: (Vec<usize>, Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>),
) -> Option<()> {
    let shape = data.0;
    let weights = data.1;
    let biases = data.2;

    let mut buf = String::new();
    buf.push('[');
    for i in 0..shape.len() {
        let size = shape[i].to_string();
        buf.push_str(&size);
        if i < shape.len() - 1 {
            //If not the last item (to avoid trailing commas)
            buf.push(',');
        }
    }
    buf.push_str(&"],\n[");

    for layer_i in 0..weights.len() {
        buf.push_str(&{
            let mut buf = String::from("[");
            for neuron_i in 0..weights[layer_i].len() {
                buf.push_str(&{
                    let mut buf = String::from("[");
                    for connection_i in 0..weights[layer_i][neuron_i].len() {
                        buf.push_str(&(weights[layer_i][neuron_i][connection_i].to_string()));
                        if connection_i < weights[layer_i][neuron_i].len() - 1 {
                            buf.push(',');
                        } else {
                            buf.push(']');
                        }
                    }
                    buf
                });
                if neuron_i < weights[layer_i].len() - 1 {
                    buf.push(',');
                } else {
                    buf.push(']');
                }
            }
            buf
        });
        if layer_i < weights.len() - 1 {
            buf.push(',');
        } else {
            buf.push(']');
        }
    }
    buf.push_str(&",\n[");

    for layer_i in 0..biases.len() {
        buf.push('[');
        buf.push_str(&{
            let mut buf = String::new();
            for neuron_i in 0..biases[layer_i].len() {
                buf.push_str(&(biases[layer_i][neuron_i].to_string()));
                if neuron_i < biases[layer_i].len() - 1 {
                    buf.push(',');
                } else {
                    buf.push(']');
                }
            }
            buf
        });
        if layer_i < biases.len() - 1 {
            buf.push(',');
        } else {
            buf.push(']');
        }
    }

    write_file(path, &buf)?;
    Some(())
}
