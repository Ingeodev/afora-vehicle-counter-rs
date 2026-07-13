// =============================================================================
// PREPROCESSING MODULE — Motor de preprocessing optimizado
// =============================================================================

pub mod engine;
pub mod lut;
pub mod scratch;

// tensor_writer.rs ya no es necesario — la lógica está especializada en scratch.rs

pub use engine::PreprocessingEngine;
