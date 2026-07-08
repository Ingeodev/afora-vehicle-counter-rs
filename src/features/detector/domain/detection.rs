#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: BoundingBox,
    pub class_id: u32,
    pub confidence: f32,
}

impl Detection {
    pub fn is_reliable(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }
}