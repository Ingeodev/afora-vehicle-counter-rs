use crate::core::afora_error::AforaError;
use crate::features::detector::Detector;
use crate::features::media_source::domain::frame_source::FrameSource;
use crate::features::pipeline::ports::pipeline::Pipeline;
use crate::features::pipeline::ports::subscriber_broadcast::SubscriberBroadcast;
use crate::features::tracker::ports::tracker::Tracker;

pub struct MultithreadedPipeline {
    media_source: Box<dyn FrameSource>,
    detector: Detector,
    tracker: Box<dyn Tracker>,
    broadcaster: Box<dyn SubscriberBroadcast>,
}

impl MultithreadedPipeline {
    pub fn new(
        media_source: Box<dyn FrameSource>, 
        detector: Detector, 
        tracker: Box<dyn Tracker>,
        broadcaster: Box<dyn SubscriberBroadcast>,
    ) -> Self {
        Self {
            media_source,
            detector,
            tracker,
            broadcaster
        }
    }
}

impl Pipeline for MultithreadedPipeline {
    fn run(&mut self) -> Result<(), AforaError> {
        Ok(())
    }
}