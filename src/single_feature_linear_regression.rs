use crate::{scalar};
use std::path::Path;
use std::io::Result;
use crate::read_csv::read_csv;
use crate::scalar::Scalar;
use crate::vector::Vector;


#[macro_export]
macro_rules! zeros {
    ($x: ident) => {
        {
            let mut vector: Vector = Vector::default();
            for _ in 0..$x {
                vector.push(scalar!(0));
            }
            vector
        }
    };
    ($x: expr) => {
        {
            let mut vector: Vector = Vector::default();
            for _ in 0..$x {
                vector.push(scalar!(0));
            }
            vector
        }
    };
    ($x: literal) => {
        {
            let mut vector: Vector = Vector::default();
            for _ in 0..$x {
                vector.push(scalar!(0));
            }
            vector
        }
    };
}


macro_rules! summa {
    ($x: ident) => {
        {
            let mut result: Scalar = scalar!(0);
            for item in $x {
                result += item;
            }
            result
        }
    };
    ($x: expr) => {
        {
            let mut result: Scalar = scalar!(0);
            for item in $x {
                result += item;
            }
            result
        }
    }
}


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

    pub fn train(&mut self) {
        for i in 0..self.iterations {
            self.bias -= self.learning_rate * self.bias_derivative();
            self.weight -= self.learning_rate * self.weight_derivative();
            if i % 100 == 0 {
                println!("Iteration {i}:\nCost: {cost}\nself: {self:?}\n", cost=self.self_cost())
            }
        }
    }

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

    /// $\frac{d}{d_x}\sum_{i=0}^{n}x_iL_i$
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
    pub fn predict(&self, x: Scalar) -> Scalar { self.weight * x  + self.bias }  // gut
    fn loss(predicted: Scalar, actual: Scalar) -> Scalar { predicted - actual }  // gut
    fn cost(predicted: Vector, actual: Vector) -> Scalar {
        assert_eq!(predicted.len(), actual.len());

        let length: usize = predicted.len();
        let mut sum: Scalar = scalar!(0);
        for i in 0..length {
            sum += Self::loss(predicted[i], actual[i]).pow(2);
        }

        sum / Scalar::from(length as u32)
    }
    fn self_cost(&self) -> Scalar {
        Self::cost(
            Vector::from(self.data.clone().into_iter().map(|x| self.predict(x)).collect::<Vec<Scalar>>()),
            self.data.clone()
        )
    }
}