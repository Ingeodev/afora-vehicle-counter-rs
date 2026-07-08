/// Placeholder mínimo de Frame — en tu crate real esto vive en `shared::domain::frame`
/// con la lógica real de resize/normalize/letterbox.
#[derive(Debug, Clone)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>, // RGB/BGR crudo
}

impl Frame {
    pub fn original_size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}