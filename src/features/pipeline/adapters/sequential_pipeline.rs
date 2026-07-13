use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use crate::core::afora_error::AforaError;
use crate::features::detector::Detector;
use crate::features::media_source::domain::frame_source::FrameSource;
use crate::features::pipeline::ports::pipeline::Pipeline;
use crate::features::pipeline::ports::subscriber_broadcast::SubscriberBroadcast;
use crate::features::tracker::domain::tracking_input::TrackingInput;
use crate::features::tracker::ports::tracker::Tracker;
use crate::features::tracking_suscribers::domain::tracking_subscriber_input::{FrameTrackingProps, TrackingSubscriberInput};
use crate::stacktrace;

pub struct SequentialPipeline {
    media_source: Box<dyn FrameSource>,
    detector: Detector,
    tracker: Box<dyn Tracker>,
    broadcaster: Box<dyn SubscriberBroadcast>,
}

impl SequentialPipeline {
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
            broadcaster,
        }
    }
}

impl Pipeline for SequentialPipeline {
    fn run(&mut self) -> Result<(), AforaError> {

        self.broadcaster
            .notify(Arc::new(TrackingSubscriberInput::StartTracking));

        loop {
            
            let batch_size = self.detector.pipeline.expected_input_shape().0 as usize;

            let mut batch = Vec::with_capacity(batch_size);

            while batch.len() < batch_size {

                match self.media_source.next() {

                    Some(frame) => {
                        batch.push(Arc::new(frame?));
                    }

                    None => break,
                }
            }

            // No quedan más frames
            if batch.is_empty() {
                break;
            }

            let detections_batch = stacktrace!("detection", "inference", 
                self.detector.detect(batch.clone())
            )?;

            for (frame, detections) in batch.into_iter().zip(detections_batch) {

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

                let subscriber_input = Arc::new(FrameTrackingProps {
                    frame_id: time.as_secs(),
                    timestamp: time,
                    frame,
                    tracks,
                });

                self.broadcaster.notify(
                    Arc::new(
                        TrackingSubscriberInput::FrameWithTracking(
                            subscriber_input,
                        ),
                    ),
                );
            }
        }

        self.broadcaster
            .notify(Arc::new(TrackingSubscriberInput::EndOfTracking));

        self.broadcaster.shutdown()
    }
}

impl Drop for SequentialPipeline {
    fn drop(&mut self) {
        let _ = self.broadcaster.shutdown();
    }
}