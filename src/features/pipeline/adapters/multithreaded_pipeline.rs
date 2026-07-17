use std::rc::Rc;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{SystemTime, UNIX_EPOCH};
use flume::{bounded, Receiver, Sender};
use std::sync::Mutex;
use crate::core::afora_error::AforaError;
use crate::features::detector::application::detector::Detector;
use crate::features::detector::ports::preprocessor::Preprocessor;
use crate::features::media_source::domain::frame_source::FrameSource;
use crate::features::pipeline::ports::pipeline::Pipeline;
use crate::features::pipeline::ports::subscriber_broadcast::SubscriberBroadcast;
use crate::features::tracker::domain::tracking_input::TrackingInput;
use crate::features::tracker::ports::tracker::Tracker;
use crate::features::tracking_suscribers::domain::tracking_subscriber_input::{FrameTrackingProps, TrackingSubscriberInput};

pub struct MultithreadedPipeline {
    media_source: Box<dyn FrameSource>,
    detector: Detector,
    tracker: Arc<Mutex<Box<dyn Tracker>>>,
    broadcaster: Arc<Mutex<Box<dyn SubscriberBroadcast>>>,
    tx:  Option<Sender<TrackingInput>>,
    rv: Receiver<TrackingInput>,
}

impl MultithreadedPipeline {
    pub fn new(
        media_source: Box<dyn FrameSource>,
        detector: Detector,
        tracker: Box<dyn Tracker>,
        broadcaster: Box<dyn SubscriberBroadcast>,
    ) -> Self {
        let (tx, rv) = bounded(32);
        Self {
            media_source,
            detector,
            tracker: Arc::new(Mutex::new(tracker)),
            broadcaster: Arc::new(Mutex::new(broadcaster)),
            tx: Some(tx),
            rv
        }
    }

    fn run_tracking(&self)  -> JoinHandle<Result<(),AforaError>> {

        let rx = self.rv.clone();
        let tracker = self.tracker.clone();
        let broadcaster = self.broadcaster.clone();

        std::thread::spawn(move || -> Result<(), AforaError> {

            while let Ok(input) = rx.recv() {

                let copy_frame = input.frame.clone();

                let tracks = {
                    let mut tracker = tracker.lock();
                    tracker.unwrap().update(input)
                }?;

                let time = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap();

                let subscriber_input = Arc::new(FrameTrackingProps {
                    frame_id: time.as_secs(),
                    timestamp: time,
                    frame: copy_frame,
                    tracks,
                });

                broadcaster
                    .lock()
                    .unwrap()
                    .notify(Arc::new(
                        TrackingSubscriberInput::FrameWithTracking(
                            subscriber_input,
                        ),
                    ));
            }

            broadcaster
                .lock()
                .unwrap()
                .notify(Arc::new(
                    TrackingSubscriberInput::EndOfTracking,
                ));

            broadcaster.lock().unwrap().shutdown().unwrap();

            Ok(())

        })
    }


}

impl Pipeline for MultithreadedPipeline {
    fn run(&mut self) -> Result<(), AforaError> {

        self.broadcaster
            .lock()
            .unwrap()
            .notify(Arc::new(TrackingSubscriberInput::StartTracking));

        let tracking_handle = self.run_tracking();

        let tx = self.tx.as_ref().unwrap();

        loop {

            let batch_size = self.detector.preprocessor.batch_size() as usize;

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

            let detections_batch = self.detector.detect(batch.clone())?;

            for (frame, detections) in batch.into_iter().zip(detections_batch) {

                tx.send(TrackingInput {
                    frame: frame.clone(),
                    detections,
                });

            }
        }

        self.broadcaster
            .lock().unwrap()
            .notify(Arc::new(TrackingSubscriberInput::EndOfTracking));

        drop(self.tx.take());

        tracking_handle.join().unwrap()?;

        self.broadcaster.lock().unwrap().shutdown()
    }
}

impl Drop for MultithreadedPipeline {
    fn drop(&mut self) {
        let _ = self.broadcaster.lock().unwrap().shutdown();
    }
}