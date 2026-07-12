use std::sync::Arc;
use crate::core::afora_error::AforaError;
use crate::features::tracking_suscribers::domain::tracking_subscriber_input::TrackingSubscriberInput;
use crate::features::tracking_suscribers::ports::tracking_subscriber::TrackingSubscriber;

pub type SubscriberBuilder = Box<dyn FnOnce() -> Result<Box<dyn TrackingSubscriber>, AforaError> + Send>;


pub trait SubscriberBroadcast {
    fn notify(&mut self, input: Arc<TrackingSubscriberInput>);
    fn shutdown(&mut self) -> Result<(), AforaError>;
}