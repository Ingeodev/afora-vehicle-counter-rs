use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use flume::Sender;
use crate::core::afora_error::AforaError;
use crate::features::detector::Detector;
use crate::features::media_source::domain::frame_source::FrameSource;
use crate::features::pipeline::ports::pipeline::Pipeline;
use crate::features::tracker::domain::tracking_input::TrackingInput;
use crate::features::tracker::ports::tracker::Tracker;
use crate::features::tracking_suscribers::domain::tracking_subscriber_input::TrackingSubscriberInput;
use crate::features::tracking_suscribers::ports::tracking_subscriber::TrackingSubscriber;

pub struct SequentialPipeline {
    media_source: Box<dyn FrameSource>,
    detector: Detector,
    tracker: Box<dyn Tracker>,
    subscriber_senders: Vec<Sender<Arc<TrackingSubscriberInput>>>,
    subscriber_threads: Vec<JoinHandle<()>>,
}

impl SequentialPipeline {
    pub fn new(
        media_source: Box<dyn FrameSource>,
        detector: Detector,
        tracker: Box<dyn Tracker>,
        subscribers: Vec<Box<dyn TrackingSubscriber>>,
    ) -> Self {

        let mut subscriber_senders = Vec::new();
        let mut subscriber_threads = Vec::new();

        for mut subscriber in subscribers {

            let (tx, rx) = flume::unbounded();
            subscriber_senders.push(tx);

            let handle = std::thread::spawn(move || {
                while let Ok(frame) = rx.recv() {
                    if let Err(err) = subscriber.on_tracking_frame(frame) {
                        eprintln!("Tracking subscriber failed: {:?}", err);
                    }
                }
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

    fn notify_new_track(&mut self, input: Arc<TrackingSubscriberInput>) {
        for sender in &self.subscriber_senders {
            let _ = sender.send(input.clone());
        }
    }
}

impl Pipeline for SequentialPipeline {
    fn run(&mut self) -> Result<(), AforaError> {

        while let Some(frame) = self.media_source.next() {
            let frame = Arc::new(frame?);

            let detections = self.detector.detect(frame.clone())?;

            let tracks = self.tracker.update(TrackingInput {
                frame: frame.clone(),
                detections,
            })?;

            let time = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|_| {
                    AforaError::PostprocessError(
                        "SystemTime before UNIX EPOCH!".into(),
                    )
                })?;

            let subscriber_input = Arc::new(TrackingSubscriberInput {
                frame_id: time.as_secs(),
                timestamp: time,
                frame,
                tracks,
            });

            self.notify_new_track(subscriber_input);
        }

        Ok(())
    }


}