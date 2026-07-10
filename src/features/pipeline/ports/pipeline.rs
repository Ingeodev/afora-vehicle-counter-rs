use crate::core::afora_error::AforaError;

pub trait Pipeline {
    fn run(&mut self) -> Result<(), AforaError>;
}