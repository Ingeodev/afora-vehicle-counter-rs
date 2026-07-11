use std::sync::Arc;
use flume::Receiver;
use crate::core::afora_error::AforaError;
use crate::features::tracking_suscribers::domain::tracking_subscriber_input::{FrameTrackingProps, TrackingSubscriberInput};

pub trait TrackingSubscriber: 'static {
    
    fn on_tracking_start(&mut self) -> Result<(), AforaError> {
        Ok(())
    }
    fn on_tracking_frame(
        &mut self,
        frame: Arc<FrameTrackingProps>,
    ) -> Result<(), AforaError>;

    fn on_tracking_finalized(&mut self) -> Result<(), AforaError> {
        Ok(())
    }

    fn notify_event(&mut self, event: Arc<TrackingSubscriberInput>) -> Result<(), AforaError> {
        match event.as_ref() {
            TrackingSubscriberInput::StartTracking => { self.on_tracking_start()},
            TrackingSubscriberInput::FrameWithTracking(frame_tracking_props) => {
                self.on_tracking_frame(frame_tracking_props.clone())
            },
            TrackingSubscriberInput::EndOfTracking => { self.on_tracking_finalized() }, 
        }
    }
    
}