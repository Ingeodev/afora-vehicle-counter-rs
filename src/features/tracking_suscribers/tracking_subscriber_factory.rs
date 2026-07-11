use std::path::PathBuf;
use crate::core::afora_error::AforaError;
use crate::features::tracking_suscribers::adapters::logger_subscriber::LoggerSubscriber;
use crate::features::tracking_suscribers::adapters::video_writer_subscriber::VideoWriterSubscriber;
use crate::features::tracking_suscribers::ports::tracking_subscriber::TrackingSubscriber;

pub enum TrackerSubscriberChoice {
    Logger,
    VideoWriter {
        output_path: PathBuf,
        width: u32,
        height: u32,
        fps: u32,
        crf: u32,
    }
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
            
            TrackerSubscriberChoice::VideoWriter {         
                output_path,
                width ,
                height,
                fps,
                crf} => Ok(Box::new(
                VideoWriterSubscriber::new(output_path, width, height, fps, crf)?
            ))
            
        }
    }
}