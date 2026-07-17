use crate::core::afora_error::AforaError;

pub enum PreprocessFallbackPolicy {
    Cpu,
    Error(String)
}