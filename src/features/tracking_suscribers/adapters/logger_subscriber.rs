use std::sync::Arc;
use crate::core::afora_error::AforaError;
use crate::features::tracking_suscribers::domain::tracking_subscriber_input::TrackingSubscriberInput;
use crate::features::tracking_suscribers::ports::tracking_subscriber::TrackingSubscriber;

pub struct LoggerSubscriber {
    
}

impl LoggerSubscriber {
    pub fn new() -> Self {
        LoggerSubscriber {}
    }
}

impl TrackingSubscriber for LoggerSubscriber {
    fn on_tracking_frame(&mut self, tracks: Arc<TrackingSubscriberInput>) -> Result<(), AforaError> {
        
        println!("{:?}", tracks.tracks);
        
        Ok(())
    }
}