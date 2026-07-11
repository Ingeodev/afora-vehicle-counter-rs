use std::sync::Arc;
use std::time::Duration;
use crate::features::tracker::domain::tracking_output::TrackingOutput;
use crate::shared::domain::frame::Frame;

pub enum TrackingSubscriberInput {

    StartTracking,

    FrameWithTracking (Arc<FrameTrackingProps>),

    EndOfTracking
}

pub struct FrameTrackingProps {
    pub(crate) frame_id: u64,
    pub(crate) timestamp: Duration,
    pub(crate) frame: Arc<Frame>,
    pub(crate) tracks: Vec<TrackingOutput>
}

