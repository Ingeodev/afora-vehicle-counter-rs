use std::path::PathBuf;
use std::sync::Arc;
use crate::core::afora_error::AforaError;
use crate::features::tracking_suscribers::domain::tracking_subscriber_input::{FrameTrackingProps};
use crate::features::tracking_suscribers::ports::tracking_subscriber::TrackingSubscriber;
use crate::features::writter::adapters::video_writter::VideoWriter;

pub struct VideoWriterSubscriber {
    video_writer: VideoWriter,
}

impl VideoWriterSubscriber {
    pub fn new(
        output_path: PathBuf,
        width: u32,
        height: u32,
        fps: u32,
        crf: u32,
    ) -> Result<Self, AforaError> {
        
        let writer = VideoWriter::new(
            output_path, width, height,
            fps,
            crf,
        )?;
        
        Ok(VideoWriterSubscriber { video_writer: writer })
    }
}

impl TrackingSubscriber for VideoWriterSubscriber {

    fn on_tracking_start(&mut self) -> Result<(), AforaError> {
        println!("Tracking started to export video");

        Ok(())
    }

    fn on_tracking_frame(&mut self, tracks: Arc<FrameTrackingProps>) -> Result<(), AforaError> {
        //println!("New frame to export video");
        
        self.video_writer.write( &tracks.frame, &tracks.tracks )?;
        Ok(())
    }

    fn on_tracking_finalized(&mut self) -> Result<(), AforaError> {
        println!("Tracking finished, exporting video");
        self.video_writer.finish()?;
        Ok(())
    }
}