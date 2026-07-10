use std::sync::Arc;
use std::time::Duration;
use crate::features::tracker::domain::tracking_output::TrackingOutput;
use crate::shared::domain::frame::Frame;

pub struct TrackingSubscriberInput {
    pub frame_id: u64,
    pub timestamp: Duration,
    pub frame: Arc<Frame>,
    pub tracks: Vec<TrackingOutput>,
}