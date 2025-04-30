use std::path::Path;
use plotly::{Plot, Scatter};
use crate::linear_regression::LinearRegression;


/// Plot the actual values and the predicted values onto a graph.  
/// The resulting chart will be exported in the form of an `HTML` page.
/// ## Parameters
/// `model: LinearRegression` - The model, which will predict the values.  
/// `path: impl AsRef<Path>` - The path, to which the `HTML` page should be exported.
pub fn plot(model: LinearRegression, path: impl AsRef<Path>) {
    let mut plot = Plot::new();
    
    // Actual values
    let x: Vec<usize> = model.x
        .rows()
        .iter()
        .enumerate()
        .map(|(i, _)| i)
        .collect();
    let y: Vec<f64> = model.y
        .values()
        .iter()
        .map(|x| x.value())
        .collect();
    plot.add_trace(Scatter::new(x, y).name("Actual"));
    
    // Predicted values
    let x: Vec<usize> = model.x
        .rows()
        .iter()
        .enumerate()
        .map(|(i, _)| i)
        .collect();
    let y: Vec<f64> = model.x
        .rows()
        .iter()
        .enumerate()
        .map(|(_, x) | model.predict(x.clone()))
        .map(|x| x.value())
        .collect();
    plot.add_trace(Scatter::new(x, y).name("Predicted"));
    
    // Export the graph
    plot.write_html(path);
}