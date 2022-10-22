/// The trait used for structs that can be fed to network training methods.
pub trait NetworkFood {
    /// Method for getting a set of data points (inputs, expected outputs) for the training dataset
    fn grab_training_data(&self) -> &Vec<(Vec<f64>, Vec<f64>)>;
    /// Method for getting a set of data points (inputs, expected outputs) for the testing dataset
    fn grab_testing_data(&self) -> &Vec<(Vec<f64>, Vec<f64>)>;
}
/// The recommended implentation of the NetworkFood trait.
#[derive(Debug)]
pub struct DataSet {
    training_data: Vec<(Vec<f64>, Vec<f64>)>,
    testing_data: Vec<(Vec<f64>, Vec<f64>)>,
}
impl NetworkFood for DataSet {
    fn grab_training_data(&self) -> &Vec<(Vec<f64>, Vec<f64>)> {
        &self.training_data
    }
    fn grab_testing_data(&self) -> &Vec<(Vec<f64>, Vec<f64>)> {
        &self.testing_data
    }
}
impl DataSet {
    pub fn new(
        training_data: Vec<(Vec<f64>, Vec<f64>)>,
        testing_data: Vec<(Vec<f64>, Vec<f64>)>,
    ) -> DataSet {
        //! Creates a dataset based off a vector of the training data points and a vector of the testing data points
        DataSet {
            training_data,
            testing_data,
        }
    }
    pub fn empty() -> DataSet {
        //! Generates an empty DataSet
        DataSet {
            training_data: vec![],
            testing_data: vec![],
        }
    }
    pub fn push_training_point(&mut self, data_point: (Vec<f64>, Vec<f64>)) {
        //! Adds a data point to the training dataset
        self.training_data.push(data_point);
    }
    pub fn push_testing_point(&mut self, data_point: (Vec<f64>, Vec<f64>)) {
        //! Adds a data point ot the testing dataset
        self.testing_data.push(data_point);
    }
    pub fn gen_from_fn<F: Fn(Vec<f64>) -> Vec<f64>>(
        func: F,
        training_points: Vec<Vec<f64>>,
        testing_points: Vec<Vec<f64>>,
    ) -> DataSet {
        //! Takes the input half of the training/testing points and a closure that is used to generate the expected outputs for all of the inputs supplied.
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
        DataSet {
            training_data,
            testing_data,
        }
    }
}
