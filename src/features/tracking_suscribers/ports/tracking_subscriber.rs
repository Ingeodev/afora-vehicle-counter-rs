use std::sync::Arc;
use flume::Receiver;
use crate::core::afora_error::AforaError;
use crate::features::tracking_suscribers::domain::tracking_subscriber_input::TrackingSubscriberInput;

pub trait TrackingSubscriber: 'static {
    fn on_tracking_frame(
        &mut self,
        frame: Arc<TrackingSubscriberInput>,
    ) -> Result<(), AforaError>;
    
}