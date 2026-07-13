use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use crate::core::afora_error::AforaError;
use crate::features::tracking_suscribers::domain::tracking_subscriber_input::{FrameTrackingProps};
use crate::features::tracking_suscribers::ports::tracking_subscriber::TrackingSubscriber;
use indicatif::{ProgressBar, ProgressStyle};

pub struct LoggerSubscriber {
    fps: Arc<AtomicUsize>,
    total_frames: Arc<AtomicUsize>,
    status: ProgressBar,
}

impl LoggerSubscriber {
    pub fn new() -> Self {

        let status = ProgressBar::new(0);

        status.set_style(
            ProgressStyle::with_template("{msg:.blue}")
                .unwrap(),
        );


        LoggerSubscriber {
            fps: Arc::new(AtomicUsize::new(0)),
            total_frames: Arc::new(AtomicUsize::new(0)),
            status,
        }
    }
}

impl TrackingSubscriber for LoggerSubscriber {

    fn on_tracking_start(&mut self) -> Result<(), AforaError> {

        println!("Tracking started");

        let fps = self.fps.clone();
        let total_frames = self.total_frames.clone();
        let status = self.status.clone();

        std::thread::spawn(move || {

            loop {

                std::thread::sleep(Duration::from_secs(1));

                let current_fps = fps.swap(0, Ordering::Relaxed);
                let total = total_frames.load(Ordering::Relaxed);

                status.set_message(format!(
                    "FPS: {} / total frames: {}",
                    current_fps,
                    total
                ));

                status.tick();
            }

        });

        Ok(())
    }

    fn on_tracking_frame(&mut self, tracks: Arc<FrameTrackingProps>) -> Result<(), AforaError> {
        let now = Instant::now();

        self.fps.fetch_add(1, Ordering::Relaxed);
        self.total_frames.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    fn on_tracking_finalized(&mut self) -> Result<(), AforaError> {
        println!("Tracking finished");
        self.status.finish_and_clear();
        Ok(())
    }
}