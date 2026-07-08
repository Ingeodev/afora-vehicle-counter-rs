use crate::core::afora_error::AforaError;
use crate::shared::domain::frame::Frame;

pub trait FrameSource: Iterator<Item = Result<Frame, AforaError>> {}

impl<T> FrameSource for T
where
    T: Iterator<Item = Result<Frame, AforaError>>,
{}