use crate::features::detector::Detector;
use crate::features::media_source::domain::frame_source::FrameSource;
use crate::features::tracker::ports::tracker::Tracker;
use crate::features::tracking_suscribers::ports::tracking_subscriber::TrackingSubscriber;

pub struct PipelineConfig {
    pub execution_mode: ExecutionMode,
    pub media_source: Box<dyn FrameSource>,
    pub detector: Detector,
    pub tracker: Box<dyn Tracker>,
    pub subscribers: Vec<Box<dyn TrackingSubscriber>>,
}

pub enum ExecutionMode {
    Sequential,
    Multithreaded
}