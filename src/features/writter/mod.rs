use std::path::Path;

use image::{Rgb, RgbImage};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;

use crate::core::afora_error::AforaError;
use crate::features::detector::domain::detection::Detection;
use crate::shared::domain::frame::Frame;

pub struct ImageWriter;

impl ImageWriter {

    pub fn write<P: AsRef<Path>>(
        &self,
        frame: &Frame,
        detections: &[Detection],
        output_path: P,
    ) -> Result<(), AforaError> {

        let mut image = RgbImage::from_raw(
            frame.width,
            frame.height,
            frame.data.clone(),
        )
            .ok_or_else(|| {
                AforaError::PostprocessError(
                    "Invalid RGB image.".into(),
                )
            })?;

        for detection in detections {

            let bbox = &detection.bbox;

            let x = bbox.x1.max(0.0).round() as i32;
            let y = bbox.y1.max(0.0).round() as i32;

            let w = (bbox.x2 - bbox.x1)
                .max(1.0)
                .round() as u32;

            let h = (bbox.y2 - bbox.y1)
                .max(1.0)
                .round() as u32;

            draw_hollow_rect_mut(
                &mut image,
                Rect::at(x, y).of_size(w, h),
                Rgb([255, 0, 0]),
            );
        }

        image
            .save(output_path)
            .map_err(|e| AforaError::PostprocessError(e.to_string()))?;

        Ok(())
    }
}