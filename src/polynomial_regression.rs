use std::io::Result;
use std::path::Path;
use crate::matrix::Matrix;
use crate::scalar::Scalar;
use crate::vector::Vector;
use crate::{scalar, vector, zeros};
use crate::plot::Plotable;
use crate::read_csv::read_csv;


#[derive(Debug)]
pub struct PolynomialRegression {
    degrees: u8,
    learning_rate: Scalar,
    iterations: u32,
    pub weights: Matrix,
    bias: Scalar,
    x: Matrix,
    y: Vector
}


impl PolynomialRegression {
    /// Create a new polynomial regression model.
    /// ## Parameters
    /// `degrees: u8` - The degree of the polynomial expression.\
    /// `learning_rate: Scalar` - How fast should the model update its parameters?\
    /// `iterations: u32` - How many times should the model optimize itself?\
    /// `x: Matrix` - The training input for the model.\
    /// `y: Vector` - The training output for the model.
    /// ## Returns
    /// A polynomial regression model, on which the user need to apply the `train()` method.
    pub fn new(degrees: u8,
               learning_rate: Scalar,
               iterations: u32,
               x: Matrix,
               y: Vector) -> Self
    {
        Self {
            degrees,
            learning_rate,
            iterations,
            weights: { 
                let mut matrix: Matrix = Matrix::default();
                for _ in 0..degrees {
                    matrix.add_row(zeros!(x.dimensions()[1]))
                }
                matrix
            },
            bias: scalar!(0),
            x,
            y
        }
    }

    /// Read in a `.csv` file and interpret its data as the input and output values
    /// ## Parameters
    /// `file: impl AsRef<Path>` - The path to the `.csv` file.\
    /// `separator: char` - The character, which separates the columns in the `.csv` file.\
    /// `degrees: u8` - The degree of the polynomial expression.
    /// `learning_rate: Scalar` - How fast should the model update its parameters?\
    /// `iterations: u32` - How many times should the model optimize itself?
    /// ## Returns
    /// A polynomial regression model, on which the user need to apply the `train()` method.
    pub fn from_csv(file: impl AsRef<Path>,
                    separator: char,
                    degrees: u8,
                    learning_rate: Scalar,
                    iterations: u32) -> Result<Self>

    {
        let csv: Vec<Vec<String>> = read_csv(file, separator)?;

        let mut x: Matrix = Matrix::default();
        let mut y: Vector = Vector::default();

        for mut line in csv {
            // Parse the Y value
            match line.pop() {
                Some(s) => match s.parse::<Scalar>() {
                    Ok(x) => y.push(x),
                    Err(_) => continue
                },
                None => continue
            };

            // Parse the X values
            let v: Vector = Vector::from(
                line
                    .iter()
                    .map(|x| {
                        x.parse::<Scalar>().expect("Unable to read .csv file") })
                    .collect::<Vec<Scalar>>()
            );
            x.add_row(v);
        }

        Ok(
            Self::new(degrees, learning_rate, iterations, x, y)
        )
    }

    /// Train the model to find the best possible parameters.
    pub fn train(&mut self) {
        for _ in 0..self.iterations {
            self.bias -= self.bias_derivative() * self.learning_rate;
            self.weights -= self.weight_derivative() * self.learning_rate;
        }
    }

    /// Derivate the cost function with respect to the weight.
    /// ## Formula
    /// <sup>&part;a</sup>&frasl;<sub>&part;C</sub> =
    /// <sup>2</sup>&frasl;<sub>n</sub>
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
    /// <sup>2</sup>&frasl;<sub>n</sub>
    /// &#x2211;<sup>n</sup><sub>i=0</sub>
    /// x<sub>i</sub>L<sub>i</sub>
    /// Where n is the amount of data that we have, x<sub>i</sub> is the current training feature,
    /// and L<sub>i</sub> is the loss for the current training feature.
    fn weight_derivative(&self) -> Matrix {
        let n: usize = self.y.len();

        let mut summa: Matrix = Matrix::default();
        for i in 0..n {
            
            let prediction: Scalar = self.predict(self.x[i].clone());
            let actual: Scalar = self.y[i];
            let error: Scalar = Self::loss(prediction, actual);

            let row: Vector = self.x[i].clone() * error;
            summa.add_row(row);
        }
        summa * (scalar!(2) / scalar!(n as u32))
    }

    fn loss(prediction: Scalar, actual: Scalar) -> Scalar {
        prediction - actual
    }

    pub fn predict(&self, x: Vector) -> Scalar {
        //println!("{:?}, {}", self.weights, self.bias);
        let mut result: Scalar = self.bias;
        for degree in 1..=self.degrees {
            for (index, weight) in self.weights[degree as usize - 1].clone().into_iter().enumerate() {
                result += weight * x[index].pow(degree as i32);
            }
        }
        result
    }
}


impl Plotable for PolynomialRegression {
    fn predicted(&self) -> Vec<f64> {
        self.x.rows().iter().map(|x| self.predict(x.clone())).map(|x| x.value()).collect()
    }
    fn actual(&self) -> Vec<f64> {
        self.y.clone().into_iter().map(|x| x.value()).collect()
    }
    fn indices(&self) -> Vec<usize> {
        self.x.rows().iter().enumerate().map(|(i, _)| i).collect()
    }
}