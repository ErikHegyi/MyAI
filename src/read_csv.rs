use std::{
    path::Path,
    io::Result,
    fs::read_to_string
};


/// Read in a `.csv` file.  
/// Warning: It does **not** read in the first line, which is
/// usually just information about what each column represents.
/// ## Parameters
/// `file: P` - The path to the file  
/// `separator: char` - The character separating the data points inside the lines
/// ## Returns
/// `Result<Vec<Vec<String>>>` - A vector, with each line inside it in the form of a vector,
/// which contains the data points.  
pub fn read_csv<P: AsRef<Path>>(file: P, separator: char) -> Result<Vec<Vec<String>>> {
    // Read in the file
    let file: String = read_to_string(file)?;
    let lines: Vec<&str> = file
        .split("\n")
        .map(|x| x.trim_end())
        .collect::<Vec<&str>>();
    
    // Interpret each line
    let mut data: Vec<Vec<String>> = Vec::new();
    for line in &lines[1..] {
        let line_data: Vec<String> = line
            .split(separator)
            .map(|x| String::from(x))
            .collect::<Vec<String>>();
        data.push(line_data);
    }
    Ok(data)
}