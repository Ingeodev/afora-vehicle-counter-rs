use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use crate::core::afora_error::AforaError;
use crate::features::tracking_suscribers::domain::tracking_subscriber_input::{FrameTrackingProps};
use crate::features::tracking_suscribers::ports::tracking_subscriber::TrackingSubscriber;
use indicatif::{ProgressBar, ProgressStyle};

pub struct LoggerSubscriber {
    fps: usize,
    frame_counter:  VecDeque<Instant>,
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
            fps: 0,
            frame_counter: VecDeque::new(),
            status
        }
    }
}

impl TrackingSubscriber for LoggerSubscriber {

    fn on_tracking_start(&mut self) -> Result<(), AforaError> {
        println!("Tracking started");

        Ok(())
    }

    fn on_tracking_frame(&mut self, tracks: Arc<FrameTrackingProps>) -> Result<(), AforaError> {
        let now = Instant::now();

        // Guardar el instante del frame actual
        self.frame_counter.push_back(now);

        // Eliminar todos los que tengan más de 1 segundo
        while let Some(front) = self.frame_counter.front() {
            if now.duration_since(*front) > Duration::from_secs(1) {
                self.frame_counter.pop_front();
            } else {
                break;
            }
        }

        // FPS del último segundo
        self.fps = self.frame_counter.len();

        self.status.set_message(format!("FPS: {}", self.fps));
        self.status.tick();

        Ok(())
    }

    fn on_tracking_finalized(&mut self) -> Result<(), AforaError> {
        println!("Tracking finished");
        self.status.finish_and_clear();
        Ok(())
    }
}