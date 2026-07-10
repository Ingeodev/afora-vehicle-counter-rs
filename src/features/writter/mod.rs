pub mod adapters;

use crate::core::afora_error::AforaError;
use crate::features::tracker::domain::tracking_output::TrackingOutput;
use crate::shared::domain::frame::Frame;
use std::path::Path;
use ab_glyph::{FontArc, PxScale};
use image::{Rgb, RgbImage};
use imageproc::{
    drawing::{draw_hollow_rect_mut, draw_text_mut},
    rect::Rect,
};
