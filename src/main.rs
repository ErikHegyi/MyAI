mod matrix;
mod scalar;
mod vector;
mod read_csv;
mod single_feature_linear_regression;
mod plot;

use crate::{
    scalar::Scalar,
    single_feature_linear_regression::SingleFeatureLinearRegression
};
use crate::plot::plot;

fn main() {
    let mut model: SingleFeatureLinearRegression = SingleFeatureLinearRegression::from_csv(
        "data.csv",
        ',',
        scalar!(0.01),
        32000
    ).unwrap();
    model.train();

    for i in 0..model.data.len() {
        println!("Predicted: {p}\nActual: {a}\n", p=model.predict(model.data[i]), a=model.values[i]);
    }
    println!("{model:?}");

    plot(model);
}

