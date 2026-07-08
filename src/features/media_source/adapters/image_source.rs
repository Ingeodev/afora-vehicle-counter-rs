use std::path::{Path, PathBuf};
use crate::core::afora_error::AforaError;
use crate::shared::domain::frame::Frame;

pub struct ImageSource {
    path: PathBuf,
    consumed: bool,
}

impl ImageSource {

    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            consumed: false,
        }
    }

}

impl Iterator for ImageSource {

    type Item = Result<Frame, AforaError>;

    fn next(&mut self) -> Option<Self::Item> {

        if self.consumed {
            return None;
        }

        self.consumed = true;

        Some(load_image_as_frame(&self.path))
    }

}

fn load_image_as_frame(
    path: &Path,
) -> Result<Frame, AforaError> {

    let image = image::open(path)
        .map_err(|e| AforaError::MediaError(e.to_string()))?;

    let rgb = image.to_rgb8();

    Ok(Frame {
        width: rgb.width(),
        height: rgb.height(),
        data: rgb.into_raw(),
    })
}