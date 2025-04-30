use std::fmt::{Display, Formatter};
use std::ops::{Add, Index, Mul, Sub};
use crate::scalar::Scalar;
use crate::{vector};
use crate::vector::Vector;


#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    values: Vec<Vector>
}


impl Matrix {
    /// Get each row of the matrix
    pub fn rows(&self) -> &Vec<Vector> {
        &self.values
    }
    
    /// Add a row to the matrix
    /// ## Parameters
    /// `row: Vector` - The row to be added.
    pub fn add_row(&mut self, row: Vector) {
        self.values.push(row);
    }
    
    /// Remove a row from the matrix, returning the removed row
    /// ## Parameters
    /// `index: usize` - The index of the row to be removed.
    pub fn pop_row(&mut self, index: usize) -> Vector {
        self.values.remove(index)
    }
    
    /// Add a scalar as the last element of a row.
    /// ## Parameters
    /// `row: usize` - The index of the row, where the value should be added.  
    /// `value: Scalar` - The scalar value to be added.
    pub fn add_to_row(&mut self, row: usize, value: Scalar) {
        self.values[row].push(value)
    }
    
    /// Create a new empty matrix with the given amount of rows.
    /// ## Parameters
    /// `rows: usize` - The amount of rows the matrix should have.
    pub fn with_rows(rows: usize) -> Self {
        let mut r: Vec<Vector> = Vec::new();
        for _ in 0..rows {
            r.push(vector!());
        }
        Self { values: r }
    }
    
    /// Get the dimensions of the matrix.
    /// ## Returns
    /// `[usize; 2]` => `[rows, columns]`
    pub fn dimensions(&self) -> [usize; 2] {
        let a: usize = self.values.len();
        let b: usize = self.values[0].len();
        [a, b]
    }
    
    /// Transpose the matrix, switching its columns and rows.
    /// ## Returns
    /// The transposed matrix.
    pub fn transpose(&self) -> Self {
        let mut matrix: Self = Self::with_rows(self.dimensions()[1]);
        
        for row in self.rows() {
            for (j, value) in row.values().iter().enumerate() {
                matrix.values[j].push(value.clone())
            }
        }
        
        matrix
    }
}


impl From<Vec<Vector>> for Matrix {
    fn from(value: Vec<Vector>) -> Self {
        Self {
            values: value
        }
    }
}

impl From<Vector> for Matrix {
    fn from(value: Vector) -> Self {
        Self {
            values: vec![value]
        }
    }
}

impl Add<Matrix> for Matrix {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        // Test if the two matrices have the same dimensions
        let dimensions: [usize; 2] = self.dimensions();
        assert_eq!(dimensions, rhs.dimensions());
        
        // Create a new matrix with the given dimensions
        let mut new: Self = Self::with_rows(self.dimensions()[0]);
        for row in 0..dimensions[0] {
            for column in 0..dimensions[1] {
                let value: Scalar = self[row][column] + rhs[row][column];
                new.add_to_row(row, value);
            }
        }
        new
    }
}


impl Add<Scalar> for Matrix {
    type Output = Self;
    fn add(self, rhs: Scalar) -> Self::Output {
        let mut new: Self = Self::default();
        for row in self.values {
            new.add_row(row.clone() + rhs);
        }
        new
    }
}


impl Sub<Scalar> for Matrix {
    type Output = Self;
    fn sub(self, rhs: Scalar) -> Self::Output {
        todo!()
    }
}


impl Sub<Matrix> for Matrix {
    type Output = Self;
    fn sub(self, rhs: Matrix) -> Self::Output {
        todo!()
    }
}


impl Mul<Scalar> for Matrix {
    type Output = Self;
    fn mul(self, rhs: Scalar) -> Self::Output {
        let mut new: Self = Self::default();
        for row in self.values {
            new.add_row(row.clone() * rhs);
        }
        new
    }
}


impl Mul<Vector> for Matrix {
    type Output = Self;
    fn mul(self, rhs: Vector) -> Self::Output {
        todo!()
    }
}


impl Mul<Matrix> for Matrix {
    type Output = Self;
    fn mul(self, rhs: Matrix) -> Self::Output {
        todo!()
    }
}


impl Default for Matrix {
    fn default() -> Self {
        Matrix {
            values: vec![]
        }
    }
}


impl Index<usize> for Matrix {
    type Output = Vector;
    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}


impl Display for Matrix {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut string: String = String::new();
        for v in self.values.iter() {
            for s in v.values() {
                string += &format!(" {s} ");
            }
            string += "\n";
        }
        write!(f, "{string}")
    }
}

#[macro_export]
macro_rules! matrix {
    () => { Matrix::default() };
    ($($x: expr), *) => {
        {
            let mut vectors: Vec<Vector> = Vec::new();
            $(
                vectors.push(
                    $x
                );
            )*
            Matrix::from(vectors)
        }
    }
}