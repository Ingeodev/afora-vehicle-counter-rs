use crate::features::detector::domain::detection::BoundingBox;

#[derive(Debug)]
pub struct TrackingOutput {
    pub id: u64,
    pub bbox: BoundingBox,
    pub class_id: u32,
    pub confidence: f32,
}