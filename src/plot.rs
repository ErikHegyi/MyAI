use std::path::Path;
use plotly::{Plot, Scatter};
use crate::single_feature_linear_regression::SingleFeatureLinearRegression;


/// Plot the actual values and the predicted values onto a graph.  
/// The resulting chart will be exported in the form of an `HTML` page.
/// ## Parameters
/// `model: LinearRegression` - The model, which will predict the values.  
/// `path: impl AsRef<Path>` - The path, to which the `HTML` page should be exported.
pub fn plot(model: SingleFeatureLinearRegression, path: impl AsRef<Path>) {
    let mut plot = Plot::new();
    
    // Actual values
    let y: Vec<f64> = model.data
        .values()
        .clone()
        .iter()
        .map(|x| x.value())
        .collect();
    let x: Vec<f64> = model.values
        .values()
        .clone()
        .iter()
        .map(|x| x.value())
        .collect();
    let trace = Scatter::new(x, y.clone()).name("Actual");
    plot.add_trace(trace);
    
    // Predicted values
    let x: Vec<f64> = model
        .data
        .clone()
        .into_iter()
        .map(|x| model.predict(x).value())
        .collect::<Vec<f64>>();
    let trace = Scatter::new(x, y).name("Predicted");
    plot.add_trace(trace);
    
    // Export the graph
    plot.write_html(path);
}