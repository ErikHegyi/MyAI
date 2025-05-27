use crate::*;


/// # Sigmoid function
/// The sigmoid function maps the values between 0 and 1.
/// ## Formula
/// sigmoid = 1 / (1 + e^-x)
pub fn sigmoid(x: Scalar) -> Scalar {
    scalar!(1) / (scalar!(1) + scalar!((-x.value()).exp()))
}
