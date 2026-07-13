use crate::core::afora_error::AforaError;
use crate::features::tracker::domain::tracking_input::TrackingInput;
use crate::features::tracker::domain::tracking_output::TrackingOutput;

pub trait Tracker: Send {
    fn update(&mut self, tracking_input: TrackingInput) -> Result<Vec<TrackingOutput>, AforaError>;
}