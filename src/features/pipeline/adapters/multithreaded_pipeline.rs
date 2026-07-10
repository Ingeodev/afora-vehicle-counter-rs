use crate::core::afora_error::AforaError;
use crate::features::detector::Detector;
use crate::features::media_source::domain::frame_source::FrameSource;
use crate::features::pipeline::ports::pipeline::Pipeline;
use crate::features::tracker::ports::tracker::Tracker;
use crate::features::tracking_suscribers::ports::tracking_subscriber::TrackingSubscriber;
use crate::features::tracking_suscribers::tracking_subscriber_factory::TrackerSubscriberChoice;

pub struct MultithreadedPipeline {
    media_source: Box<dyn FrameSource>,
    detector: Detector,
    tracker: Box<dyn Tracker>,
    subscribers: Vec<TrackerSubscriberChoice>
}

impl MultithreadedPipeline {
    pub fn new(
        media_source: Box<dyn FrameSource>, 
        detector: Detector, 
        tracker: Box<dyn Tracker>,
        subscribers: Vec<TrackerSubscriberChoice>
    ) -> Self {
        Self {
            media_source,
            detector,
            tracker,
            subscribers
        }
    }
}

impl Pipeline for MultithreadedPipeline {
    fn run(&mut self) -> Result<(), AforaError> {
        Ok(())
    }
}