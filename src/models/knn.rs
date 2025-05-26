use crate::scalar::*;
use crate::matrix::Matrix;
use crate::scalar;
use crate::vector::Vector;


pub struct KNearestNeighbors {
    k: usize,
    x: Matrix,
    y: Vector
}


impl KNearestNeighbors {
    pub fn new(k: usize, x: Matrix, y: Vector) -> Self {
        Self { k, x, y }
    }
    pub fn predict(&self, value: Vector) -> Scalar {
        // Calculate the distance from each value
        let mut distances: Vec<[Scalar; 2]> = Vec::with_capacity(self.y.len());
        for (vec, y) in std::iter::zip(
            self.x.rows(),
            self.y.clone().into_iter()
        ) {
            distances.push(
                [Self::distance(&value, vec), y]
            )
        }

        // Get the K nearest neighbors
        let mut nearest: Vec<[Scalar; 2]> = Vec::with_capacity(self.k);
        for distance in distances {
            // If there aren't K elements in the vec, then just add it
            if nearest.len() < self.k { nearest.push(distance); }
            else {
                for (n, near) in nearest.clone().iter().enumerate() {
                    if near[0] > distance[0] {
                        nearest[n] = distance;
                    }
                }
            }
            
            // Sort the vec
            for a in 0..nearest.len() {
                for b in 0..nearest.len() {
                    if nearest[a] > nearest[b] {
                        (nearest[b], nearest[a]) = (nearest[a], nearest[b]);
                    }
                }
            }
        }

        let mut values: Vec<Scalar> = vec![scalar!(0); self.y.len()];
        for near in nearest {
            values[self.y.find(near[1]).unwrap()] += scalar!(1) / near[0];
        }
        
        let mut result: Scalar = self.y[0];
        let mut max: Scalar = values[0];
        for (index, value) in self.y.clone().into_iter().enumerate() {
            if values[index] > max {
                max = values[index];
                result = value;
            }
        }
        result
    }
    fn distance(a: &Vector, b: &Vector) -> Scalar {
        let mut distance: Scalar = scalar!(0);
        for (x, y) in std::iter::zip(a.clone().into_iter(), b.clone().into_iter()) {
            distance += (x - y).pow(2);
        }
        distance.sqrt()
    }
}


#[macro_export]
macro_rules! knn {
    ($k: expr, $x: expr, $y: expr) => {
        KNearestNeighbors::new($k as usize, $x, $y)
    };
    ($k: expr; $($x: expr),+; $($y: expr),+) => {
        knn!(
            $k,
            {
                let mut matrix: crate::matrix::Matrix = crate::matrix::Matrix::default();
                $(
                    matrix.add_row($x);
                )+
                matrix
            },
            {
                let mut vector: crate::vector::Vector = crate::vector::Vector::default();
                $(
                    vector.push(scalar!($y));                     
                )+
                vector
            }
        )
    };
}