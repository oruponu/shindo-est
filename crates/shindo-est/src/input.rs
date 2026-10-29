//! Inputs to a prediction.

/// An earthquake source.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hypocenter {
    /// Latitude in degrees.
    pub latitude: f64,
    /// Longitude in degrees.
    pub longitude: f64,
    /// Depth in km. NaN means the depth is unknown.
    pub depth: f64,
    /// JMA magnitude (Mj). NaN means the magnitude is unknown.
    pub magnitude: f64,
}

/// A place where the intensity is predicted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Site {
    /// Latitude in degrees.
    pub latitude: f64,
    /// Longitude in degrees.
    pub longitude: f64,
    /// Amplification of peak ground velocity relative to engineering bedrock.
    /// Must be finite and positive.
    pub arv: f64,
}
