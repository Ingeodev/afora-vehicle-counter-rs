use crate::core::afora_error::AforaError;
use crate::features::tracking_suscribers::adapters::logger_subscriber::LoggerSubscriber;
use crate::features::tracking_suscribers::ports::tracking_subscriber::TrackingSubscriber;

pub enum TrackerSubscriberChoice {
    Logger
}

pub struct TrackerSubscriberFactory;

impl TrackerSubscriberFactory {
    pub fn build(
        subscriber_choice: TrackerSubscriberChoice,
    ) -> Result<Box<dyn TrackingSubscriber>, AforaError> {
        match subscriber_choice {
            TrackerSubscriberChoice::Logger => Ok(Box::new(
                LoggerSubscriber::new()
            )),
        }
    }
}