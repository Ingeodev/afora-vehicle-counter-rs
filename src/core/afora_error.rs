use std::fmt;

#[derive(Debug)]
pub enum AforaError {
    /// El modelo requiere capacidades que el runtime no soporta
    IncompatibleCombination {
        runtime: &'static str,
        model: &'static str,
        missing_capabilities: Vec<String>,
    },
    /// El shape que el pipeline espera no coincide con el spec real del modelo cargado
    ShapeMismatch {
        expected: (u32, u32, u32),
        actual: Vec<i64>,
    },
    /// Error al cargar el modelo en el runtime (archivo corrupto, backend no disponible, etc.)
    RuntimeLoadError(String),
    /// Error durante la ejecución de inferencia
    InferenceError(String),
    /// Error durante preprocesamiento del frame
    PreprocessError(String),
    /// Error durante postprocesamiento de la salida cruda
    PostprocessError(String),
    /// El runtime solicitado no fue compilado (feature flag desactivado)
    RuntimeNotCompiled(&'static str),
    
    MediaError(String),
    InvalidArgument(String),
}



impl fmt::Display for AforaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AforaError::IncompatibleCombination { runtime, model, missing_capabilities } => {
                write!(
                    f,
                    "'{model}' requiere {missing_capabilities:?}, que el runtime '{runtime}' no soporta"
                )
            }
            AforaError::ShapeMismatch { expected, actual } => {
                write!(f, "shape esperado {expected:?} no coincide con el shape real del modelo {actual:?}")
            }
            AforaError::RuntimeLoadError(msg) => write!(f, "error al cargar runtime: {msg}"),
            AforaError::InferenceError(msg) => write!(f, "error de inferencia: {msg}"),
            AforaError::PreprocessError(msg) => write!(f, "error de preprocesamiento: {msg}"),
            AforaError::PostprocessError(msg) => write!(f, "error de postprocesamiento: {msg}"),
            AforaError::RuntimeNotCompiled(name) => {
                write!(f, "el runtime '{name}' no fue compilado (feature flag desactivado)")
            }
            AforaError::MediaError(msg) => write!(f, "Error al cargar la el recurso multimedia {msg}"),
            AforaError::InvalidArgument(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for AforaError {}