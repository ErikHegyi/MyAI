use crate::{scalar};
use std::path::Path;
use std::io::Result;
use crate::read_csv::read_csv;
use crate::scalar::Scalar;
use crate::vector::Vector;


#[derive(Debug)]
pub struct SingleFeatureLinearRegression {
    learning_rate: Scalar,
    iterations: u32,
    weight: Scalar,
    bias: Scalar,
    pub data: Vector,
    pub values: Vector
}

impl SingleFeatureLinearRegression {
    /// Create a new linear regression model.
    /// ## Parameters
    /// `learning_rate: f64` - The learning rate of the model (how fast should it update its parameters?)  
    /// `iterations: u32` - How many times should it run itself?  
    /// `data: Vector` - The data that the program should train itself on - the **input values**.  
    /// `values: Vector` - The data that the program should train itself on - the **output values**.  
    /// ## Returns
    /// `Self` - An instance of a linear regression model
    pub fn new(learning_rate: Scalar, iterations: u32, data: Vector, values: Vector) -> Self {
        Self {
            learning_rate,
            iterations,
            weight: scalar!(0),
            bias: scalar!(0),
            data,
            values
        }
    }
    
    /// Create a new linear regression model by reading in a `.csv` file.
    /// ## Parameters
    /// `file: P` - The path to the `.csv` file  
    /// `separator: char` - The character separating the data points in the file  
    /// `learning_rate: f64` - The learning rate of the model  
    /// `iterations: u32` - How many times should the model fit itself
    /// ## Returns
    /// `Self` - An instance of a linear regression model
    pub fn from_csv<P: AsRef<Path>>(
        file: P,
        separator: char,
        learning_rate: Scalar,
        iterations: u32
    ) -> Result<Self> {
        let csv: Vec<Vec<String>> = read_csv(file, separator)?;
        
        let mut data: Vector = Vector::default();
        let mut values: Vector = Vector::default();
        
        // Interpret each line
        for line in csv {
            let line_data: &String = line.first().expect("Unable to read in .csv file.");
            let line_value: &String = line.last().expect("Unable to read in .csv file.");

            // Interpret the training data (x)
            let line_data: Scalar = Scalar::from({
                if line_data.contains('.') {
                    Scalar::from(line_data.parse::<f64>().unwrap())
                } else {
                    Scalar::from(line_data.parse::<i32>().unwrap())
                }
            });

            // Interpret the result (y)
            let line_value: Scalar = match line_value.contains('.') {
                true => Scalar::from(line_value.parse::<f64>().unwrap()),
                false => Scalar::from(line_value.parse::<i32>().unwrap())
            };

            data.push(line_data);
            values.push(line_value);
        }
        Ok(Self::new(learning_rate, iterations, data, values))
    }
    
    /// Train the model to find the best possible parameters.
    pub fn train(&mut self) {
        for _ in 0..self.iterations {
            self.bias -= self.learning_rate * self.bias_derivative();
            self.weight -= self.learning_rate * self.weight_derivative();
        }
    }

    /// Derivate the cost function with respect to the weight.
    /// ## Formula
    /// <sup>&part;a</sup>&frasl;<sub>&part;C</sub> = 
    /// <sup>1</sup>&frasl;<sub>n</sub>
    /// &#x2211;<sup>n</sup><sub>i=0</sub>
    /// L<sub>i</sub>  
    /// Where n is the amount of data that we have and L<sub>i</sub> is the loss 
    /// for the current training feature.
    fn bias_derivative(&self) -> Scalar {
        let n: usize = self.values.len();

        let mut summa: Scalar = scalar!(0);
        for i in 0..n {
            let prediction: Scalar = self.predict(self.data[i]);
            let actual: Scalar = self.values[i];
            let error: Scalar = Self::loss(prediction, actual);
            summa += error;
        }
        scalar!(1) / scalar!(n as u32 as f64) * summa
    }
    
    /// Derivate the cost function with respect to the weight.
    /// ## Formula
    /// <sup>&part;a</sup>&frasl;<sub>&part;C</sub> = 
    /// <sup>1</sup>&frasl;<sub>n</sub>
    /// &#x2211;<sup>n</sup><sub>i=0</sub>
    /// x<sub>i</sub>L<sub>i</sub>  
    /// Where n is the amount of data that we have, x<sub>i</sub> is the current training feature,
    /// and L<sub>i</sub> is the loss for the current training feature.
    fn weight_derivative(&self) -> Scalar {
        let n: usize = self.values.len();

        let mut summa: Scalar = scalar!(0);
        for i in 0..n {
            let prediction: Scalar = self.predict(self.data[i]);
            let actual: Scalar = self.values[i];
            let error: Scalar = Self::loss(prediction, actual);
            summa += error * self.data[i];
        }
        scalar!(1) / scalar!(n as u32 as f64) * summa
    }
    
    /// Predict an output value based on input `x`.
    /// ## Formula
    /// `y = ax + b`  
    /// Where `y` is the output value, `a` is the weight of the function, `x` is the input value
    /// and `b` is the bias.
    pub fn predict(&self, x: Scalar) -> Scalar {
        self.weight * x  + self.bias
    }
    
    /// Calculate the loss based on the predicted and actual values.
    /// ## Formula
    /// L = y<sub>predicted</sub> - y<sub>actual</sub>
    fn loss(predicted: Scalar, actual: Scalar) -> Scalar {
        predicted - actual
    }
    
    /// Calculate the cost (loss) for the current training data with the current parameters.
    /// ## Formula
    /// C = <sup>1</sup>&frasl;<sub>n</sub>
    /// &#x2211;<sup>n</sup><sub>i=0</sub>L<sub>i</sub><sup>2</sup>  
    /// Where n is the amount of training data and L<sub>i</sub> is the loss for the
    /// current data.
    fn cost(predicted: Vector, actual: Vector) -> Scalar {
        // The two vectors have to be of the same size
        let n: usize = predicted.len();
        assert_eq!(n, actual.len());

        let mut sum: Scalar = scalar!(0);
        for i in 0..n {
            sum += Self::loss(predicted[i], actual[i]).pow(2);
        }

        sum / Scalar::from(n as u32)
    }
    
    /// Calculate the value of the cost function with the training data and
    /// the current predictions for each element of the training data.
    fn self_cost(&self) -> Scalar {
        let actual: Vector = self.data.clone();
        let predicted: Vector = Vector::from(
            self.data
                .clone()
                .into_iter()
                .map(|x| self.predict(x))
                .collect::<Vec<Scalar>>()
        );
        
        Self::cost(predicted, actual)
    }
}