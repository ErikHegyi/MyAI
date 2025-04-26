use std::fmt::{Display, Formatter};
use std::ops::{Add, Index, Mul, Sub};
use std::io::{ErrorKind, Result, Error};
use crate::scalar;
use crate::scalar::Scalar;


#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub struct Vector {
    values: Vec<Scalar>
}


impl Vector {
    pub fn values(&self) -> &Vec<Scalar> {
        &self.values
    }
    
    pub fn push(&mut self, value: Scalar) {
        self.values.push(value)
    } 
    
    pub fn pop(&mut self, index: usize) -> Scalar {
        self.values.remove(index)
    }
    
    pub fn find(&self, value: Scalar) -> Option<Scalar> {
        self.values.iter().find(|x| **x == value).and_then(|x| Some(*x))
    }
    
    pub fn len(&self) -> usize {
        self.values.len()
    }
    pub fn sum(&self) -> Scalar {
        let mut s: Scalar = scalar!(0);
        for value in self.values() {
            s += value.clone();
        }
        s
    }
}


impl From<&[Scalar]> for Vector {
    fn from(value: &[Scalar]) -> Self {
        let mut new: Self = Self { values: Vec::new() };
        for v in value {
            new.values.push(*v)
        }
        new
    }
}


impl<P: std::fmt::Debug> From<Vec<P>> for Vector where Scalar: From<P> {
    fn from(value: Vec<P>) -> Self {
        let mut values: Vec<Scalar> = Vec::new();
        for v in value {
            values.push(
                Scalar::from(v)
            );
        }
        
        Self { values }
    }
}



impl Default for Vector {
    fn default() -> Self {
        Vector { values: Vec::new() }
    }
}


impl Add<Vector> for Vector {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        // Test if the lengths are equal
        assert_eq!(self.values.len(), rhs.values.len());
        
        // Add the two vectors together
        let mut new: Vector = Vector::default();
        for i in 0..self.values.len() {
            new.values.push(self.values[i] + rhs.values[i]);
        }
        new
    }
}

impl Add<Scalar> for Vector {
    type Output = Self;
    fn add(self, rhs: Scalar) -> Self::Output {
        let mut new: Vector = self.clone();
        for value in new.values.iter_mut() {
           *value += rhs;
        }
        new
    }
}


impl Sub for Vector {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        // Test if the lengths are equal
        assert_eq!(self.values.len(), rhs.values.len());

        // Subtract the second vector from the first vector
        let mut new: Vector = Vector::default();
        for i in 0..self.values.len() {
            new.values.push(self.values[i] - rhs.values[i]);
        }
        new
    }
}


impl Mul<Vector> for Vector {
    type Output = Scalar;
    fn mul(self, rhs: Self) -> Self::Output {
        // Test if the lengths are equal
        assert_eq!(self.values.len(), rhs.values.len());

        // Multiply the two vectors together (dot product)
        let mut result: Scalar = Scalar::default();
        for i in 0..self.values.len() {
            result += self[i] * rhs[i];
        }
        result
    }
}


impl Mul<Scalar> for Vector {
    type Output = Self;
    fn mul(self, rhs: Scalar) -> Self::Output {
        let mut new: Vector = self.clone();
        for value in new.values.iter_mut() {
            *value *= rhs;
        }
        new
    }
}


impl Index<usize> for Vector {
    type Output = Scalar;
    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}


impl IntoIterator for Vector {
    type Item = Scalar;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}


impl Display for Vector {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.values)
    }
}


#[macro_export]
/// Create a new vector from the given elements
macro_rules! vector {
    () => { Vector::default() };
    ($($x: literal),*) => {
        {
            let mut vec: Vec<Scalar> = Vec::new();
            $(
                vec.push(
                    scalar!($x)
                );
            )*
            Vector::from(vec)
        }
    };
}
