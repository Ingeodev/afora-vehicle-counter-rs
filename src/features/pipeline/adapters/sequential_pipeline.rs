use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use flume::{unbounded, Sender};
use crate::core::afora_error::AforaError;
use crate::features::detector::Detector;
use crate::features::media_source::domain::frame_source::FrameSource;
use crate::features::pipeline::ports::pipeline::Pipeline;
use crate::features::tracker::domain::tracking_input::TrackingInput;
use crate::features::tracker::ports::tracker::Tracker;
use crate::features::tracking_suscribers::domain::tracking_subscriber_input::{FrameTrackingProps, TrackingSubscriberInput};
use crate::features::tracking_suscribers::ports::tracking_subscriber::TrackingSubscriber;
use crate::features::tracking_suscribers::tracking_subscriber_factory::{TrackerSubscriberChoice, TrackerSubscriberFactory};

pub struct SequentialPipeline {
    media_source: Box<dyn FrameSource>,
    detector: Detector,
    tracker: Box<dyn Tracker>,
    subscriber_senders: Vec<Sender<Arc<TrackingSubscriberInput>>>,
    subscriber_threads: Vec<JoinHandle<Result<(), AforaError>>>,
}

impl SequentialPipeline {
    pub fn new(
        media_source: Box<dyn FrameSource>,
        detector: Detector,
        tracker: Box<dyn Tracker>,
        subscribers: Vec<TrackerSubscriberChoice>,
    ) -> Self {

        let mut subscriber_senders = Vec::new();
        let mut subscriber_threads = Vec::new();

        for mut subscriber_choice in subscribers {

            let (tx, rx) = unbounded();
            subscriber_senders.push(tx);


            let handle = std::thread::spawn(move || -> Result<(), AforaError> {

                let mut subscriber = TrackerSubscriberFactory::build(subscriber_choice)?;
                
                while let Ok(frame) = rx.recv() {
                    subscriber.notify_event(frame)?
                }
                
                Ok(())
            });
            subscriber_threads.push(handle);
        }

        Self {
            media_source,
            detector,
            tracker,
            subscriber_senders,
            subscriber_threads,
        }
    }

    fn notify_event(&mut self, input: Arc<TrackingSubscriberInput>) {
        for sender in &self.subscriber_senders {
            let _ = sender.send(input.clone());
        }
    }
    
}

impl Pipeline for SequentialPipeline {
    fn run(&mut self) -> Result<(), AforaError> {
        
        let event_start = Arc::new(TrackingSubscriberInput::StartTracking);
        let event_end = Arc::new(TrackingSubscriberInput::EndOfTracking);
        
        self.notify_event(event_start);
        
        
        while let Some(frame) = self.media_source.next() {
            let frame = Arc::new(frame?);

            let detections = self.detector.detect(frame.clone())?;

            let tracks = self.tracker.update(TrackingInput {
                frame: frame.clone(),
                detections,
            })?;

            //TODO: Cambiar esto por los datos reales desde la construccion del frame
            let time = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|_| {
                    AforaError::PostprocessError(
                        "SystemTime before UNIX EPOCH!".into(),
                    )
                })?;

            let subscriber_input = Arc::new(FrameTrackingProps {
                frame_id: time.as_secs(),
                timestamp: time,
                frame,
                tracks,
            });
            
            let event = Arc::new(TrackingSubscriberInput::FrameWithTracking(subscriber_input));

            self.notify_event(event);
        }
        
        self.notify_event(event_end);

        Ok(())
    }


}