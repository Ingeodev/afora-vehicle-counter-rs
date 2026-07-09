use std::sync::Arc;
use crate::features::detector::domain::detection::Detection;
use crate::shared::domain::frame::Frame;

pub struct TrackingInput {
    pub frame: Arc<Frame>,
    pub detections: Vec<Detection>,
}