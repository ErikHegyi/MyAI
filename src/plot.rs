use plotly::{Plot, Scatter};
use crate::single_feature_linear_regression::SingleFeatureLinearRegression;

pub fn plot(model: SingleFeatureLinearRegression) {
    println!("{:?}", model.values);
    let mut plot = Plot::new();
    let y = model.data.values().clone().iter().map(|x| x.value()).collect();
    let x = model.values.values().clone().iter().map(|x| x.value()).collect();
    let trace = Scatter::new(x, y).name("Actual");
    plot.add_trace(trace);

        println!("{:?}", model.values);

    let y = model.data.values().clone().iter().map(|x| x.value()).collect();
    let x = model
        .data
        .clone()
        .into_iter()
        .map(|x| model.predict(x).value())
        .collect::<Vec<f64>>();

    let trace = Scatter::new(x, y).name("Predicted");
    plot.add_trace(trace);
    plot.write_html("hello.html");
}