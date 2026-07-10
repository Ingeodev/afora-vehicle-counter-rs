

use ab_glyph::FontArc;
use image::RgbImage;
use std::path::Path;

use crate::core::afora_error::AforaError;
use crate::features::tracker::domain::tracking_output::TrackingOutput;
use crate::shared::domain::frame::Frame;
use crate::shared::domain::overlay::{draw_overlays, load_default_font};

pub struct ImageWriter {
    font: FontArc,
}

impl ImageWriter {
    pub fn new() -> Result<Self, AforaError> {
        Ok(Self { font: load_default_font()? })
    }

    pub fn write<P: AsRef<Path>>(
        &self,
        frame: &Frame,
        tracks: &[TrackingOutput],
        output_path: P,
    ) -> Result<(), AforaError> {
        let mut image = RgbImage::from_raw(frame.width, frame.height, frame.data.clone())
            .ok_or_else(|| AforaError::PostprocessError("Invalid RGB image.".into()))?;

        draw_overlays(&mut image, tracks, &self.font);

        image
            .save(output_path)
            .map_err(|e| AforaError::PostprocessError(e.to_string()))?;
        Ok(())
    }
}