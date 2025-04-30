use std::path::Path;
use plotly::{Plot, Scatter};


pub trait Plotable {
    fn predicted(&self) -> Vec<f64>;
    fn actual(&self) -> Vec<f64>;
    fn indices(&self) -> Vec<usize>;
}


/// Plot the actual values and the predicted values onto a graph.  
/// The resulting chart will be exported in the form of an `HTML` page.
/// ## Parameters
/// `model: LinearRegression` - The model, which will predict the values.  
/// `path: impl AsRef<Path>` - The path, to which the `HTML` page should be exported.
pub fn plot(model: impl Plotable, path: impl AsRef<Path>) {
    let mut plot = Plot::new();
    
    // Actual values
    let x: Vec<usize> = model.indices();
    let y: Vec<f64> = model.actual();
    plot.add_trace(Scatter::new(x, y).name("Actual"));
    
    // Predicted values
    let x: Vec<usize> = model.indices();
    let y: Vec<f64> = model.predicted();
    plot.add_trace(Scatter::new(x, y).name("Predicted"));
    
    // Export the graph
    plot.write_html(path);
}