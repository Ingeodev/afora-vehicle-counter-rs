use std::path::{ PathBuf};
use crate::core::afora_error::AforaError;
use crate::features::media_source::adapters::image_source::ImageSource;
use crate::features::media_source::adapters::video_source::VideoSource;
use crate::features::media_source::domain::frame_source::FrameSource;

pub enum MediaSourceChoice {
    Image{path: PathBuf},
    Video{path: PathBuf, max_frames: Option<i32>},
}

pub struct MediaSourceFactory;

impl MediaSourceFactory {
    pub fn build(
        src: MediaSourceChoice,
    ) -> Result<Box<dyn FrameSource>, AforaError> {
        match src {
            MediaSourceChoice::Image{path} => Ok(Box::new(
                ImageSource::new(path),
            )),
            MediaSourceChoice::Video{path, max_frames} => {
                let mut source = VideoSource::new(path)?;
                if let Some(frames) = max_frames {
                    source = source.with_max_frame_limit(frames)
                }
                Ok(Box::new(source))
            }
        }
    }
}