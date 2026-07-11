use std::sync::Arc;
use crate::core::afora_error::AforaError;
use crate::features::tracking_suscribers::domain::tracking_subscriber_input::{FrameTrackingProps};
use crate::features::tracking_suscribers::ports::tracking_subscriber::TrackingSubscriber;

pub struct LoggerSubscriber {
    
}

impl LoggerSubscriber {
    pub fn new() -> Self {
        LoggerSubscriber {}
    }
}

impl TrackingSubscriber for LoggerSubscriber {

    fn on_tracking_start(&mut self) -> Result<(), AforaError> {
        println!("Tracking started");

        Ok(())
    }

    fn on_tracking_frame(&mut self, tracks: Arc<FrameTrackingProps>) -> Result<(), AforaError> {
        println!("{:?}", tracks.tracks);
        
        Ok(())
    }

    fn on_tracking_finalized(&mut self) -> Result<(), AforaError> {
        println!("Tracking finished");

        Ok(())
    }
}