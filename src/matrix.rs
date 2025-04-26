use std::fmt::{Display, Formatter};
use std::ops::{Add, Index, Mul};
use crate::scalar::Scalar;
use crate::{vector};
use crate::vector::Vector;


#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    values: Vec<Vector>
}


impl Matrix {
    pub fn rows(&self) -> &Vec<Vector> {
        &self.values
    }
    pub fn add_row(&mut self, row: Vector) {
        self.values.push(row);
    }
    pub fn pop_row(&mut self, index: usize) -> Vector {
        self.values.remove(index)
    }
    pub fn add_to_row(&mut self, row: usize, value: Scalar) {
        self.values[row].push(value)
    }
    pub fn with_rows(rows: usize) -> Self {
        let mut r: Vec<Vector> = Vec::new();
        for _ in 0..rows {
            r.push(vector!());
        }
        Self { values: r }
    }
    pub fn dimensions(&self) -> [usize; 2] {
        let a: usize = self.values.len();
        let b: usize = self.values[0].len();
        [a, b]
    }
    pub fn transpose(&self) -> Self {
        let mut matrix: Self = Self::with_rows(self.dimensions()[1]);
        for row in self.rows() {
            for (j, value) in row.values().iter().enumerate() {
                matrix.values[j].push(value.clone())
            }
        }
        matrix
    }
    pub fn dot(&self, rhs: Self) -> Self {
        // Test if the matrices are compatible
        let dimensions: [usize; 2] = self.dimensions();
        println!("self.dimensions = {dimensions:?}\nself = [{self}]\nrhs.dimensions = {rhs_dim:?}\nrhs = [{rhs}]", rhs_dim=rhs.dimensions());
        assert_eq!(dimensions[0], rhs.dimensions()[1]);
        
        // Transpose the other matrix
        let t: Matrix = rhs.transpose();
        
        // Construct the new matrix
        let mut new: Self = Self::with_rows(dimensions[0]);
        for (i, row) in new.values.iter_mut().enumerate() {
            row.push(
                self.values[i].clone() * t.values[i].clone()
            ); 
        }
        new
    }
    pub fn to_vector(&self) -> Vector {
        let dimensions: [usize; 2] = self.dimensions();
        
        if dimensions[0] == 1 { self.values[0].clone() }
        else if dimensions[1] == 1 {
            let mut vector: Vector = Vector::default();
            for row in self.rows() {
                vector.push(row[0])
            }
            vector
        } else {
            panic!("The matrix is not one dimensional")
        }
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


impl Default for Matrix {
    fn default() -> Self {
        Matrix {
            values: vec![vector!()]
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