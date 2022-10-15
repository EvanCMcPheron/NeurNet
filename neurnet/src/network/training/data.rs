pub trait NetworkFood {
    fn grab_training_data(&self) -> &Vec<(Vec<f64> ,Vec<f64>)>;
    fn grab_testing_data(&self) -> &Vec<(Vec<f64> ,Vec<f64>)>;
}
#[derive(Debug)]
pub struct DataSet {
    training_data: Vec<(Vec<f64>, Vec<f64>)>,
    testing_data: Vec<(Vec<f64>, Vec<f64>)>,
}
impl NetworkFood for DataSet {
    fn grab_training_data(&self) -> &Vec<(Vec<f64> ,Vec<f64>)> {
        &self.training_data
    }
    fn grab_testing_data(&self) -> &Vec<(Vec<f64> ,Vec<f64>)> {
        &self.testing_data
    }
}
impl DataSet {
    pub fn new(training_data: Vec<(Vec<f64>, Vec<f64>)>, testing_data: Vec<(Vec<f64>, Vec<f64>)>) -> DataSet {
        DataSet {
            training_data,
            testing_data,
        }
    }
    pub fn empty() -> DataSet {
        DataSet { training_data: vec![], testing_data: vec![] }
    }
    pub fn push_training_point(&mut self, data_point: (Vec<f64>, Vec<f64>)) {
        self.training_data.push(data_point);
    }
    pub fn push_testing_point(&mut self, data_point: (Vec<f64>, Vec<f64>)) {
        self.testing_data.push(data_point);
    }
    pub fn gen_from_fn<F: Fn(Vec<f64>) -> Vec<f64>>(func: F, training_points: Vec<Vec<f64>>, testing_points: Vec<Vec<f64>>) -> DataSet {
        let training_data = {
            let mut buf: Vec<(Vec<f64>, Vec<f64>)> = vec![];
            for x in training_points {
                buf.push((x.clone(), func(x)))
            }
            buf
        };
        let testing_data = {
            let mut buf: Vec<(Vec<f64>, Vec<f64>)> = vec![];
            for x in testing_points {
                buf.push((x.clone(), func(x)))
            }
            buf
        };
        DataSet { training_data, testing_data }
    }
}