use std::f64;
use std::fmt::{Debug, Display, Formatter};
use std::num::{ParseFloatError, ParseIntError};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use thiserror::Error;
use std::str::FromStr;


#[derive(Error, Debug)]
pub enum ScalarError {
    #[error("Could not parse float into scalar")]
    ParseFloatError(#[from] ParseFloatError),
    #[error("Could not parse integer into scalar")]
    ParseIntError(#[from] ParseIntError)
}

#[derive(Clone, Copy, PartialOrd, PartialEq)]
pub struct Scalar {
    value: f64 
}


impl Scalar {
    /// Get the value of the scalar in the form of a floating-point integer.
    /// ## Returns
    /// `f64` - The value of the scalar.
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Set the value of the scalar.
    /// ## Parameters
    /// `value: f64` - The new value of the scalar.
    pub fn set_value(&mut self, value: f64) {
        self.value = value;
    }

    /// Raise the scalar to the given power.
    /// ## Parameters
    /// `n: i32` - The exponent of the power.
    /// ## Returns
    /// `Scalar` - The scalar raised to the power of `n`.
    pub fn pow(&self, n: i32) -> Self {
        Self::from(self.value.powi(n))
    }
    
    /// Get the square root of the given number.
    /// ## Returns
    /// `Scalar` - The square root of the scalar.
    pub fn sqrt(&self) -> Self {
        Self::from(self.value.sqrt())
    }

    /// Get the absolute value of the scalar.
    /// ## Returns
    /// `Self` - The absolute value of the scalar.
    pub fn abs(&self) -> Self {
        Self::from(self.value.abs())
    }
}


impl<P> From<P> for Scalar where f64: From<P> {
    fn from(value: P) -> Self {
        Self { value: f64::from(value) }
    }
}


impl Display for Scalar {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.3}", self.value)
    }
}


impl Debug for Scalar {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}


impl Add for Scalar {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value + rhs.value
        }
    }
}


impl Sub for Scalar {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value - rhs.value
        }
    }
}


impl Mul for Scalar {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value * rhs.value
        }
    }
}


impl Div for Scalar {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value / rhs.value
        }
    }
}


impl AddAssign for Scalar {
    fn add_assign(&mut self, rhs: Self) {
        *self = Self { value: self.value + &rhs.value }
    }
}


impl SubAssign for Scalar {
    fn sub_assign(&mut self, rhs: Self) {
        *self = Self { value: self.value - &rhs.value }
    }
}


impl MulAssign for Scalar {
    fn mul_assign(&mut self, rhs: Self) {
        *self = Self { value: self.value * &rhs.value }
    }
}


impl DivAssign for Scalar {
    fn div_assign(&mut self, rhs: Self) {
        *self = Self { value: self.value / &rhs.value }
    }
}


impl Neg for Scalar {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            value: -self.value
        }
    }
}

impl FromStr for Scalar {
    type Err = ScalarError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.contains('.') {
            true => match s.parse::<f64>() {
                Ok(x) => Ok(Scalar::from(x)),
                Err(e) => Err(ScalarError::ParseFloatError(e))
            },
            false => match s.parse::<i32>() {
                Ok(x) => Ok(Scalar::from(x)),
                Err(e) => Err(ScalarError::ParseIntError(e))
            }
        }
    }
}


impl Default for Scalar {
    fn default() -> Self {
        Self { value: f64::default() }
    }
}


#[macro_export]
macro_rules! scalar {
    ($x: literal) => {
        Scalar::from($x)
    };
    ($x: ident) => {
        Scalar::from($x)
    };
    ($x: expr) => {
        Scalar::from($x)
    }
}