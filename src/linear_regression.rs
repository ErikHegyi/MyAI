use crate::{scalar, vector};
use std::path::Path;
use std::io::Result;
use crate::matrix::Matrix;
use crate::read_csv::read_csv;
use crate::scalar::Scalar;
use crate::vector::Vector;


#[macro_export]
macro_rules! zeros {
    ($x: literal) => {
        {
            let mut vector: Vector = vector!();
            for _ in 0..$x {
                vector.push(scalar!(0))
            }
            vector
        }
    };
    ($x: ident) => {
        {
            let mut vector: Vector = vector!();
            for _ in 0..$x {
                vector.push(scalar!(0))
            }
            vector
        }
    };
    ($x: expr) => {
        {
            let mut vector: Vector = vector!();
            for _ in 0..$x {
                vector.push(scalar!(0))
            }
            vector
        }
    };
}

#[derive(Debug)]
pub struct LinearRegression {
    learning_rate: Scalar,
    iterations: u32,
    weights: Vector,
    bias: Scalar,
    pub x: Matrix,
    pub y: Vector
}

impl LinearRegression {
    /// Create a new linear regression model.
    /// ## Parameters
    /// `learning_rate: f64` - The learning rate of the model (how fast should it update its parameters?)  
    /// `iterations: u32` - How many times should it run itself?  
    /// `data: Matrix` - The data that the program should train itself on - the **input values**.
    /// `values: Vector` - The data that the program should train itself on - the **output values**.  
    /// ## Returns
    /// `Self` - An instance of a linear regression model
    pub fn new(learning_rate: Scalar, iterations: u32, data: Matrix, values: Vector) -> Self {
        Self {
            learning_rate,
            iterations,
            weights: zeros!(data.dimensions()[1]),
            bias: scalar!(0),
            x: data,
            y: values
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
        
        let mut data: Matrix = Matrix::default();
        let mut values: Vector = Vector::default();
        
        // Interpret each line
        for mut line in csv {
            if line.len() < 2 { continue; }
            let line_value: String = line.remove(line.len() - 1);

            // Interpret the training data (x)
            let line_data: Vector = Vector::from({
                line
                    .iter()
                    .map(
                        |x| {
                            if x.contains('.') {
                                x.parse::<f64>().expect("Unable to parse .csv file.")
                            } else {
                                x.parse::<i32>().expect("Unable to parse .csv file.") as f64
                            }
                        }
                    )
                    .collect::<Vec<f64>>()
            });

            // Interpret the result (y)
            let line_value: Scalar = match line_value.contains('.') {
                true => Scalar::from(line_value.parse::<f64>().unwrap()),
                false => Scalar::from(line_value.parse::<i32>().unwrap())
            };

            data.add_row(line_data);
            values.push(line_value);
        }
        Ok(Self::new(learning_rate, iterations, data, values))
    }
    
    /// Train the model to find the best possible parameters.
    pub fn train(&mut self) {
        for _ in 0..self.iterations {
            self.bias -= self.learning_rate * self.bias_derivative();
            self.weights -= self.weight_derivative() * self.learning_rate;
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
        let n: usize = self.y.len();

        let mut summa: Scalar = scalar!(0);
        for i in 0..n {
            let prediction: Scalar = self.predict(self.x[i].clone());
            let actual: Scalar = self.y[i];
            let error: Scalar = Self::loss(prediction, actual);
            summa += error;
        }
        scalar!(2) / scalar!(n as u32 as f64) * summa
    }
    
    /// Derivate the cost function with respect to the weight.
    /// ## Formula
    /// <sup>&part;a</sup>&frasl;<sub>&part;C</sub> = 
    /// <sup>1</sup>&frasl;<sub>n</sub>
    /// &#x2211;<sup>n</sup><sub>i=0</sub>
    /// x<sub>i</sub>L<sub>i</sub>  
    /// Where n is the amount of data that we have, x<sub>i</sub> is the current training feature,
    /// and L<sub>i</sub> is the loss for the current training feature.
    fn weight_derivative(&self) -> Vector {
        let n: usize = self.y.len();

        let mut summa: Vector = zeros!(self.x.dimensions()[1]);
        for i in 0..n {
            let prediction: Scalar = self.predict(self.x[i].clone());
            let actual: Scalar = self.y[i];
            let error: Scalar = Self::loss(prediction, actual);

            summa += self.x[i].clone() * error;
        }
        summa * scalar!(2) / scalar!(n as u32 as f64)
    }
    
    /// Predict an output value based on input `x`.
    /// ## Formula
    /// `y = ax + b`  
    /// Where `y` is the output value, `a` is the weight of the function, `x` is the input value
    /// and `b` is the bias.
    pub fn predict(&self, x: Vector) -> Scalar {
        self.weights.clone() * x  + self.bias
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
        // Clone the actual Y values
        let actual: Vector = self.y.clone();
        
        // Predict the values
        let predicted: Vector = {
            // Create an empty vector
            let mut v: Vector = vector!();
            
            // Iterate through each row in the X matrix
            for vec in self.x.rows() {
                let prediction: Scalar = self.predict(vec.clone());                
                v.push(prediction);
            }
            v
        };
        
        Self::cost(predicted, actual)
    }
}